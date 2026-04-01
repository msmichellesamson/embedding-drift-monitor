# Redis and PostgreSQL for embedding drift monitoring

resource "google_redis_instance" "embedding_cache" {
  name               = "${var.project_name}-redis"
  tier               = "STANDARD_HA"
  memory_size_gb     = 4
  region             = var.region
  redis_version      = "REDIS_7_0"
  display_name       = "Embedding Cache"
  
  authorized_network = google_compute_network.vpc.id
  
  maintenance_policy {
    weekly_maintenance_window {
      day = "SUNDAY"
      start_time {
        hours = 2
        minutes = 0
        seconds = 0
      }
    }
  }
  
  labels = {
    environment = var.environment
    component   = "cache"
  }
}

resource "google_sql_database_instance" "postgres" {
  name             = "${var.project_name}-db"
  database_version = "POSTGRES_15"
  region          = var.region
  deletion_protection = false
  
  settings {
    tier                        = "db-custom-2-8192"
    deletion_protection_enabled = false
    
    database_flags {
      name  = "shared_preload_libraries"
      value = "pg_stat_statements"
    }
    
    backup_configuration {
      enabled    = true
      start_time = "03:00"
      location   = var.region
      
      point_in_time_recovery_enabled = true
      transaction_log_retention_days  = 7
    }
    
    ip_configuration {
      ipv4_enabled    = false
      private_network = google_compute_network.vpc.id
      require_ssl     = true
    }
    
    insights_config {
      query_insights_enabled  = true
      query_plans_per_minute  = 5
      query_string_length     = 1024
      record_application_tags = true
    }
  }
  
  depends_on = [google_service_networking_connection.private_vpc_connection]
}

resource "google_sql_database" "embedding_drift" {
  name     = "embedding_drift"
  instance = google_sql_database_instance.postgres.name
}

resource "google_sql_user" "app_user" {
  name     = "drift_monitor"
  instance = google_sql_database_instance.postgres.name
  password = var.db_password
}

resource "google_service_networking_connection" "private_vpc_connection" {
  network                 = google_compute_network.vpc.id
  service                 = "servicenetworking.googleapis.com"
  reserved_peering_ranges = [google_compute_global_address.private_ip_address.name]
}

resource "google_compute_global_address" "private_ip_address" {
  name          = "${var.project_name}-private-ip"
  purpose       = "VPC_PEERING"
  address_type  = "INTERNAL"
  prefix_length = 16
  network       = google_compute_network.vpc.id
}

output "redis_host" {
  value = google_redis_instance.embedding_cache.host
}

output "postgres_connection" {
  value = google_sql_database_instance.postgres.connection_name
  sensitive = true
}