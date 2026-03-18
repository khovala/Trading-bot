terraform {
  required_version = ">= 1.5.0"

  required_providers {
    yandex = {
      source  = "yandex-cloud/yandex"
      version = "~> 0.130.0"
    }
  }

  backend "s3" {
    endpoint   = "https://storage.yandexcloud.net"
    bucket     = "moex-trading-terraform"
    key        = "terraform/state"
    region     = "ru-central1"
    access_key = var.yc_access_key
    secret_key = var.yc_secret_key
  }
}

provider "yandex" {
  cloud_id  = var.yc_cloud_id
  folder_id = var.yc_folder_id
  zone      = "ru-central1-a"

  service_account_key_file = var.yc_service_account_key
}

variable "yc_cloud_id" {
  description = "Yandex Cloud ID"
  type        = string
  sensitive   = true
}

variable "yc_folder_id" {
  description = "Yandex Cloud Folder ID"
  type        = string
  sensitive   = true
}

variable "yc_service_account_key" {
  description = "Path to service account key JSON"
  type        = string
}

variable "yc_access_key" {
  description = "S3 access key for Object Storage"
  type        = string
  sensitive   = true
}

variable "yc_secret_key" {
  description = "S3 secret key for Object Storage"
  type        = string
  sensitive   = true
}

variable "ssh_public_key" {
  description = "SSH public key for VM access"
  type        = string
  default     = "~/.ssh/id_rsa.pub"
}

variable "db_password" {
  description = "PostgreSQL database password"
  type        = string
  sensitive   = true
}

locals {
  prefix = "moex-trading"
  tags   = {
    project     = "moex-sandbox"
    environment = "production"
    managed_by  = "terraform"
  }
}

resource "yandex_vpc_network" "trading_network" {
  name = "${local.prefix}-network"
  tags = values(local.tags)
}

resource "yandex_vpc_subnet" "trading_subnet" {
  name           = "${local.prefix}-subnet"
  network_id     = yandex_vpc_network.trading_network.id
  v4_cidr_blocks = ["10.128.0.0/24"]
  zone           = "ru-central1-a"
}

resource "yandex_compute_instance" "trading_vm" {
  name        = "${local.prefix}-vm"
  platform_id = "standard-v3"
  zone        = "ru-central1-a"

  resources {
    cores  = 4
    memory = 8
    core_fraction = 50
  }

  boot_disk {
    initialize_params {
      image_id = "ubuntu-22.04-lts"
      size     = 100
      type     = "network-nvme"
    }
  }

  network_interface {
    subnet_id  = yandex_vpc_subnet.trading_subnet.id
    nat        = true
    ip_version = "ipv4"
  }

  metadata = {
    ssh-keys = "ubuntu:${file(var.ssh_public_key)}"
    docker-compose = file("${path.module}/docker-compose.yml")
  }

  connection {
    type        = "ssh"
    user        = "ubuntu"
    private_key = file("~/.ssh/id_rsa")
    host        = self.network_interface[0].nat_ip_address
  }

  provisioner "file" {
    source      = ".env"
    destination = "/home/ubuntu/.env"
  }

  provisioner "remote-exec" {
    inline = [
      "sudo apt-get update && sudo apt-get install -y docker.io docker-compose",
      "sudo systemctl enable docker",
      "sudo usermod -aG docker ubuntu",
      "mkdir -p /opt/trading",
    ]
  }

  tags = values(local.tags)
}

resource "yandex_mdb_postgresql_cluster" "trading_db" {
  name        = "${local.prefix}-postgres"
  environment = "PRODUCTION"
  network_id  = yandex_vpc_network.trading_network.id

  config {
    version = "16"
    resources {
      resource_preset_id = "s2.medium"
      disk_type          = "network-ssd"
      disk_size          = 20
    }
  }

  database {
    name = "trading"
  }

  user {
    name     = "trading_user"
    password = var.db_password
    permission {
      database_name = "trading"
    }
  }

  host {
    zone       = "ru-central1-a"
    subnet_id  = yandex_vpc_subnet.trading_subnet.id
    assign_public_ip = true
  }

  maintenance_window {
    type = "WEEKLY"
    day  = "SUN"
    hour = 3
  }

  tags = values(local.tags)
}

resource "yandex_storage_bucket" "mlflow_artifacts" {
  bucket = "moex-trading-mlflow"
  acl    = "private"

  anonymous_access_blocks = true

  versioning {
    enabled = true
  }

  tags = values(local.tags)
}

resource "yandex_storage_bucket" "terraform_state" {
  bucket = "moex-trading-terraform"
  acl    = "private"

  anonymous_access_blocks = true

  tags = values(local.tags)
}

resource "yandex_iam_service_account" "vm_agent" {
  name = "${local.prefix}-vm-agent"
  description = "Service account for trading VM"
}

resource "yandex_resourcemanager_folder_iam_binding" "vm_agent_binding" {
  folder_id = var.yc_folder_id
  role      = "editor"
  members   = ["serviceAccount:${yandex_iam_service_account.vm_agent.id}"]
}

resource "yandex_compute_instance_group" "trading_ig" {
  name               = "${local.prefix}-ig"
  folder_id          = var.yc_folder_id
  network_id         = yandex_vpc_network.trading_network.id
  instance_template {
    platform_id = "standard-v3"
    resources {
      cores  = 4
      memory = 8
    }
    boot_disk {
      initialize_params {
        image_id = "ubuntu-22.04-lts"
        size     = 50
      }
    }
    network_interface {
      subnet_ids = [yandex_vpc_subnet.trading_subnet.id]
      nat        = true
    }
    metadata = {
      ssh-keys = "ubuntu:${file(var.ssh_public_key)}"
    }
  }

  scale_policy {
    fixed_scale {
      size = 1
    }
  }

  allocation_policy {
    zones = ["ru-central1-a"]
  }

  deploy_policy {
    max_unavailable = 1
    max_creating     = 1
  }

  tags = values(local.tags)
}

output "vm_public_ip" {
  value = yandex_compute_instance.trading_vm.network_interface[0].nat_ip_address
}

output "postgres_connection_string" {
  value     = "postgresql://trading_user:${var.db_password}@${yandex_mdb_postgresql_cluster.trading_db.host[0].fqdn}:5432/trading"
  sensitive = true
}

output "storage_bucket" {
  value = yandex_storage_bucket.mlflow_artifacts.bucket
}
