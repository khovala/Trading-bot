variable "yc_cloud_id" {
  description = "Yandex Cloud ID"
  type        = string
}

variable "yc_folder_id" {
  description = "Yandex Cloud Folder ID"
  type        = string
}

variable "yc_service_account_key" {
  description = "Path to service account key JSON"
  type        = string
  default     = "./service-account-key.json"
}

variable "yc_access_key" {
  description = "S3 access key for Object Storage"
  type        = string
}

variable "yc_secret_key" {
  description = "S3 secret key for Object Storage"
  type        = string
}

variable "ssh_public_key" {
  description = "SSH public key for VM access"
  type        = string
  default     = "~/.ssh/id_rsa.pub"
}

variable "db_password" {
  description = "PostgreSQL database password"
  type        = string
}
