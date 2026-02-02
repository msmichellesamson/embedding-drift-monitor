terraform {
  required_version = ">= 1.0"
  required_providers {
    google = {
      source  = "hashicorp/google"
      version = "~> 5.0"
    }
    kubernetes = {
      source  = "hashicorp/kubernetes"
      version = "~> 2.23"
    }
    helm = {
      source  = "hashicorp/helm"
      version = "~> 2.11"
    }
  }
}

variable "project_id" {
  description = "GCP project ID"
  type        = string
}

variable "region" {
  description = "GCP region"
  type        = string
  default     = "us-central1"
}

variable "zone" {
  description = "GCP zone"
  type        = string
  default     = "us-central1-a"
}

variable "environment" {
  description = "Environment (dev, staging, prod)"
  type        = string
  default     = "dev"
}

variable "cluster_name" {
  description = "GKE cluster name"
  type        = string
  default     = "embedding-drift-monitor"
}

variable "db_instance_name" {
  description = "Cloud SQL instance name"
  type        = string
  default     = "embedding-drift-db"
}

variable "redis_instance_name" {
  description = "Redis instance name"
  type        = string
  default     = "embedding-drift-cache"
}

variable "node_count" {
  description = "Initial number of nodes in the GKE cluster"
  type        = number
  default     = 3
}

variable "machine_type" {
  description = "Machine type for GKE nodes"
  type        = string
  default     = "e2-standard-4"
}

locals {
  common_labels = {
    environment = var.environment
    project     = "embedding-drift-monitor"
    managed_by  = "terraform"
  }
}

provider "google" {
  project = var.project_id
  region  = var.region
}

# Enable required APIs
resource "google_project_service" "required_apis" {
  for_each = toset([
    "container.googleapis.com",
    "sqladmin.googleapis.com",
    "redis.googleapis.com",
    "monitoring.googleapis.com",
    "logging.googleapis.com",
    "cloudresourcemanager.googleapis.com",
    "compute.googleapis.com"
  ])
  
  project = var.project_id
  service = each.value
  
  disable_dependent_services = false
  disable_on_destroy        = false
}

# VPC Network
resource "google_compute_network" "vpc" {
  name                    = "${var.cluster_name}-vpc"
  auto_create_subnetworks = false
  
  depends_on = [google_project_service.required_apis]
}

# Subnet for GKE
resource "google_compute_subnetwork" "gke_subnet" {
  name          = "${var.cluster_name}-subnet"
  ip_cidr_range = "10.0.0.0/20"
  region        = var.region
  network       = google_compute_network.vpc.name
  
  secondary_ip_range {
    range_name    = "gke-pods"
    ip_cidr_range = "10.1.0.0/16"
  }
  
  secondary_ip_range {
    range_name    = "gke-services"
    ip_cidr_range = "10.2.0.0/16"
  }
}

# Private subnet for Cloud SQL and Redis
resource "google_compute_subnetwork" "private_subnet" {
  name          = "${var.cluster_name}-private-subnet"
  ip_cidr_range = "10.3.0.0/24"
  region        = var.region
  network       = google_compute_network.vpc.name
  
  private_ip_google_access = true
}

# VPC Peering for private services
resource "google_compute_global_address" "private_ip_range" {
  name          = "${var.cluster_name}-private-ip"
  purpose       = "VPC_PEERING"
  address_type  = "INTERNAL"
  prefix_length = 20
  network       = google_compute_network.vpc.id
  
  depends_on = [google_project_service.required_apis]
}

resource "google_service_networking_connection" "private_vpc_connection" {
  network                 = google_compute_network.vpc.id
  service                 = "servicenetworking.googleapis.com"
  reserved_peering_ranges = [google_compute_global_address.private_ip_range.name]
  
  depends_on = [google_project_service.required_apis]
}

# GKE Cluster
resource "google_container_cluster" "primary" {
  name     = var.cluster_name
  location = var.region
  
  remove_default_node_pool = true
  initial_node_count       = 1
  
  network    = google_compute_network.vpc.name
  subnetwork = google_compute_subnetwork.gke_subnet.name
  
  ip_allocation_policy {
    cluster_secondary_range_name  = "gke-pods"
    services_secondary_range_name = "gke-services"
  }
  
  private_cluster_config {
    enable_private_nodes    = true
    enable_private_endpoint = false
    master_ipv4_cidr_block  = "10.4.0.0/28"
  }
  
  network_policy {
    enabled = true
  }
  
  workload_identity_config {
    workload_pool = "${var.project_id}.svc.id.goog"
  }
  
  addons_config {
    http_load_balancing {
      disabled = false
    }
    
    horizontal_pod_autoscaling {
      disabled = false
    }
    
    network_policy_config {
      disabled = false
    }
  }
  
  master_auth {
    client_certificate_config {
      issue_client_certificate = false
    }
  }
  
  resource_labels = local.common_labels
  
  depends_on = [
    google_project_service.required_apis,
    google_compute_subnetwork.gke_subnet
  ]
}

# GKE Node Pool
resource "google_container_node_pool" "primary_nodes" {
  name       = "${var.cluster_name}-nodes"
  location   = var.region
  cluster    = google_container_cluster.primary.name
  node_count = var.node_count
  
  node_config {
    preemptible  = false
    machine_type = var.machine_type
    disk_size_gb = 100
    disk_type    = "pd-ssd"
    
    service_account = google_service_account.gke_service_account.email
    
    oauth_scopes = [
      "https://www.googleapis.com/auth/cloud-platform",
      "https://www.googleapis.com/auth/logging.write",
      "https://www.googleapis.com/auth/monitoring",
      "https://www.googleapis.com/auth/devstorage.read_only"
    ]
    
    workload_metadata_config {
      mode = "GKE_METADATA"
    }
    
    labels = local.common_labels
    
    tags = ["gke-node", "${var.cluster_name}-node"]
  }
  
  management {
    auto_repair  = true
    auto_upgrade = true
  }
  
  autoscaling {
    min_node_count = 1
    max_node_count = 10
  }
  
  upgrade_settings {
    max_surge       = 2
    max_unavailable = 1
  }
  
  depends_on = [google_service_account.gke_service_account]
}

# Service Account for GKE nodes
resource "google_service_account" "gke_service_account" {
  account_id   = "${var.cluster_name}-gke-sa"
  display_name = "GKE Service Account for ${var.cluster_name}"
}

# IAM bindings for GKE service account
resource "google_project_iam_member" "gke_service_account_roles" {
  for_each = toset([
    "roles/logging.logWriter",
    "roles/monitoring.metricWriter",
    "roles/monitoring.viewer",
    "roles/stackdriver.resourceMetadata.writer",
    "roles/storage.objectViewer",
    "roles/cloudsql.client",
    "roles/redis.editor"
  ])
  
  project = var.project_id
  role    = each.value
  member  = "serviceAccount:${google_service_account.gke_service_account.email}"
}

# Cloud SQL Instance
resource "google_sql_database_instance" "postgres" {
  name             = "${var.db_instance_name}-${var.environment}"
  database_version = "POSTGRES_15"
  region           = var.region
  
  settings {
    tier              = "db-custom-2-7680"
    availability_type = var.environment == "prod" ? "REGIONAL" : "ZONAL"
    disk_size         = 100
    disk_type         = "PD_SSD"
    disk_autoresize   = true
    
    backup_configuration {
      enabled                        = true
      start_time                     = "03:00"
      point_in_time_recovery_enabled = true
      transaction_log_retention_days = 7
      backup_retention_settings {
        retained_backups = 30
      }
    }
    
    maintenance_window {
      day          = 7
      hour         = 4
      update_track = "stable"
    }
    
    ip_configuration {
      ipv4_enabled                                  = false
      private_network                               = google_compute_network.vpc.id
      enable_private_path_for_google_cloud_services = true
    }
    
    database_flags {
      name  = "max_connections"
      value = "200"
    }
    
    database_flags {
      name  = "shared_preload_libraries"
      value = "pg_stat_statements"
    }
    
    database_flags {
      name  = "log_statement"
      value = "all"
    }
    
    database_flags {
      name  = "log_min_duration_statement"
      value = "1000"
    }
    
    insights_config {
      query_insights_enabled  = true
      query_string_length     = 1024
      record_application_tags = true
      record_client_address   = true
    }
    
    user_labels = local.common_labels
  }
  
  deletion_protection = var.environment == "prod" ? true : false
  
  depends_on = [
    google_service_networking_connection.private_vpc_connection,
    google_project_service.required_apis
  ]
}

# Cloud SQL Database
resource "google_sql_database" "embedding_drift_db" {
  name     = "embedding_drift"
  instance = google_sql_database_instance.postgres.name
}

# Cloud SQL User
resource "google_sql_user" "app_user" {
  name     = "app_user"
  instance = google_sql_database_instance.postgres.name
  password = random_password.db_password.result
}

resource "random_password" "db_password" {
  length  = 32
  special = true
}

# Redis Instance
resource "google_redis_instance" "cache" {
  name           = "${var.redis_instance_name}-${var.environment}"
  tier           = "STANDARD_HA"
  memory_size_gb = 4
  region         = var.region
  
  location_id             = var.zone
  alternative_location_id = "${substr(var.region, 0, length(var.region) - 1)}b"
  
  authorized_network = google_compute_network.vpc.id
  connect_mode       = "PRIVATE_SERVICE_ACCESS"
  
  redis_version     = "REDIS_7_0"
  display_name      = "Embedding Drift Cache ${var.environment}"
  reserved_ip_range = "10.5.0.0/29"
  
  redis_configs = {
    maxmemory-policy = "allkeys-lru"
    notify-keyspace-events = "Ex"
  }
  
  maintenance_policy {
    weekly_maintenance_window {
      day = "SATURDAY"
      start_time {
        hours   = 4
        minutes = 0
        seconds = 0
        nanos   = 0
      }
    }
  }
  
  labels = local.common_labels
  
  depends_on = [
    google_service_networking_connection.private_vpc_connection,
    google_project_service.required_apis
  ]
}

# Kubernetes Secret for Database
resource "kubernetes_secret" "db_secret" {
  metadata {
    name      = "postgres-secret"
    namespace = "default"
    labels    = local.common_labels
  }
  
  data = {
    POSTGRES_HOST     = google_sql_database_instance.postgres.private_ip_address
    POSTGRES_PORT     = "5432"
    POSTGRES_DB       = google_sql_database.embedding_drift_db.name
    POSTGRES_USER     = google_sql_user.app_user.name
    POSTGRES_PASSWORD = google_sql_user.app_user.password
    DATABASE_URL      = "postgresql://${google_sql_user.app_user.name}:${google_sql_user.app_user.password}@${google_sql_database_instance.postgres.private_ip_address}:5432/${google_sql_database.embedding_drift_db.name}"
  }
  
  type = "Opaque"
  
  depends_on = [google_container_node_pool.primary_nodes]
}

# Kubernetes Secret for Redis
resource "kubernetes_secret" "redis_secret" {
  metadata {
    name      = "redis-secret"
    namespace = "default"
    labels    = local.common_labels
  }
  
  data = {
    REDIS_HOST = google_redis_instance.cache.host
    REDIS_PORT = tostring(google_redis_instance.cache.port)
    REDIS_URL  = "redis://${google_redis_instance.cache.host}:${google_redis_instance.cache.port}"
  }
  
  type = "Opaque"
  
  depends_on = [google_container_node_pool.primary_nodes]
}

# Kubernetes provider configuration
provider "kubernetes" {
  host  = "https://${google_container_cluster.primary.endpoint}"
  token = data.google_client_config.current.access_token
  
  cluster_ca_certificate = base64decode(
    google_container_cluster.primary.master_auth[0].cluster_ca_certificate,
  )
}

provider "helm" {
  kubernetes {
    host  = "https://${google_container_cluster.primary.endpoint}"
    token = data.google_client_config.current.access_token
    
    cluster_ca_certificate = base64decode(
      google_container_cluster.primary.master_auth[0].cluster_ca_certificate,
    )
  }
}

data "google_client_config" "current" {}

# Prometheus Operator via Helm
resource "helm_release" "prometheus_operator" {
  name       = "prometheus-operator"
  repository = "https://prometheus-community.github.io/helm-charts"
  chart      = "kube-prometheus-stack"
  namespace  = "monitoring"
  version    = "51.2.0"
  
  create_namespace = true
  
  values = [
    yamlencode({
      grafana = {
        enabled = true
        adminPassword = random_password.grafana_password.result
        service = {
          type = "LoadBalancer"
        }
        persistence = {
          enabled = true
          size = "10Gi"
        }
      }
      
      prometheus = {
        prometheusSpec = {
          retention = "30d"
          storageSpec = {
            volumeClaimTemplate = {
              spec = {
                storageClassName = "standard-rwo"
                accessModes = ["ReadWriteOnce"]
                resources = {
                  requests = {
                    storage = "50Gi"
                  }
                }
              }
            }
          }
          serviceMonitorSelectorNilUsesHelmValues = false
          podMonitorSelectorNilUsesHelmValues = false
          ruleSelectorNilUsesHelmValues = false
        }
      }
      
      alertmanager = {
        alertmanagerSpec = {
          storage = {
            volumeClaimTemplate = {
              spec = {
                storageClassName = "standard-rwo"
                accessModes = ["ReadWriteOnce"]
                resources = {
                  requests = {
                    storage = "10Gi"
                  }
                }
              }
            }
          }
        }
      }
    })
  ]
  
  depends_on = [google_container_node_pool.primary_nodes]
}

resource "random_password" "grafana_password" {
  length  = 16
  special = true
}

# Kubernetes Secret for Grafana password
resource "kubernetes_secret" "grafana_secret" {
  metadata {
    name      = "grafana-admin-secret"
    namespace = "default"
    labels    = local.common_labels
  }
  
  data = {
    GRAFANA_ADMIN_PASSWORD = random_password.grafana_password.result
  }
  
  type = "Opaque"
  
  depends_on = [google_container_node_pool.primary_nodes]
}

# Outputs
output "cluster_name" {
  description = "GKE cluster name"
  value       = google_container_cluster.primary.name
}

output "cluster_endpoint" {
  description = "GKE cluster endpoint"
  value       = google_container_cluster.primary.endpoint
  sensitive   = true
}

output "cluster_location" {
  description = "GKE cluster location"
  value       = google_container_cluster.primary.location
}

output "postgres_connection_name" {
  description = "Cloud SQL connection name"
  value       = google_sql_database_instance.postgres.connection_name
}

output "postgres_private_ip" {
  description = "Cloud SQL private IP"
  value       = google_sql_database_instance.postgres.private_ip_address
  sensitive   = true
}

output "redis_host" {
  description = "Redis instance host"
  value       = google_redis_instance.cache.host
  sensitive   = true
}

output "redis_port" {
  description = "Redis instance port"
  value       = google_redis_instance.cache.port
}

output "grafana_password" {
  description = "Grafana admin password"
  value       = random_password.grafana_password.result
  sensitive   = true
}

output "kubectl_config_command" {
  description = "Command to configure kubectl"
  value       = "gcloud container clusters get-credentials ${google_container_cluster.primary.name} --region ${google_container_cluster.primary.location} --project ${var.project_id}"
}