resource "google_container_cluster" "embedding_drift_cluster" {
  name     = "embedding-drift-monitor"
  location = var.region
  
  # Use autopilot mode for managed infrastructure
  enable_autopilot = true
  
  # Network configuration
  network    = google_compute_network.vpc.name
  subnetwork = google_compute_subnetwork.subnet.name
  
  # Security
  master_auth {
    client_certificate_config {
      issue_client_certificate = false
    }
  }
  
  # Workload Identity
  workload_identity_config {
    workload_pool = "${var.project_id}.svc.id.goog"
  }
  
  # Monitoring
  monitoring_config {
    enable_components = ["SYSTEM_COMPONENTS", "WORKLOADS"]
    managed_prometheus {
      enabled = true
    }
  }
  
  # Logging
  logging_config {
    enable_components = ["SYSTEM_COMPONENTS", "WORKLOADS"]
  }
  
  # Networking
  ip_allocation_policy {
    cluster_secondary_range_name  = "pods"
    services_secondary_range_name = "services"
  }
  
  # Maintenance
  maintenance_policy {
    daily_maintenance_window {
      start_time = "03:00"
    }
  }
  
  depends_on = [
    google_compute_network.vpc,
    google_compute_subnetwork.subnet
  ]
}

# Service account for workload identity
resource "google_service_account" "gke_workload" {
  account_id   = "gke-workload-sa"
  display_name = "GKE Workload Service Account"
}

# IAM binding for workload identity
resource "google_service_account_iam_binding" "workload_identity" {
  service_account_id = google_service_account.gke_workload.name
  role               = "roles/iam.workloadIdentityUser"
  
  members = [
    "serviceAccount:${var.project_id}.svc.id.goog[default/embedding-drift-monitor]"
  ]
}