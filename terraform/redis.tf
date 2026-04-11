# Redis cluster for embedding cache and drift metrics
resource "google_redis_instance" "drift_cache" {
  name           = "embedding-drift-cache"
  memory_size_gb = 4
  region         = var.region
  tier           = "STANDARD_HA"
  
  redis_version     = "REDIS_6_X"
  display_name      = "Embedding Drift Cache"
  reserved_ip_range = "10.0.0.0/29"
  
  auth_enabled               = true
  transit_encryption_mode    = "SERVER_AUTHENTICATION"
  authorized_network         = google_compute_network.drift_monitor_vpc.id
  connect_mode              = "PRIVATE_SERVICE_ACCESS"
  
  maintenance_policy {
    weekly_maintenance_window {
      day = "SUNDAY"
      start_time {
        hours   = 2
        minutes = 0
      }
    }
  }
  
  labels = {
    environment = var.environment
    service     = "embedding-drift-monitor"
    component   = "cache"
  }
}

# Output Redis connection details
output "redis_host" {
  value = google_redis_instance.drift_cache.host
}

output "redis_port" {
  value = google_redis_instance.drift_cache.port
}

output "redis_auth_string" {
  value     = google_redis_instance.drift_cache.auth_string
  sensitive = true
}