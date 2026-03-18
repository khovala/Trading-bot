output "vm_public_ip" {
  value = yandex_compute_instance.trading_vm.network_interface[0].nat_ip_address
  description = "Public IP address of the trading VM"
}

output "postgres_hosts" {
  value = yandex_mdb_postgresql_cluster.trading_db.host
  description = "PostgreSQL cluster hosts"
}

output "storage_bucket_name" {
  value = yandex_storage_bucket.mlflow_artifacts.bucket
  description = "S3 bucket name for MLflow artifacts"
}
