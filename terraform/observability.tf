# Observability Stack
resource "google_container_cluster" "observability" {
  name     = "observability-stack"
  location = var.region

  remove_default_node_pool = true
  initial_node_count       = 1

  network    = google_compute_network.main.name
  subnetwork = google_compute_subnetwork.main.name
}

resource "google_container_node_pool" "observability_nodes" {
  name       = "observability-nodes"
  location   = var.region
  cluster    = google_container_cluster.observability.name
  node_count = 2

  node_config {
    preemptible  = true
    machine_type = "e2-standard-2"
    disk_size_gb = 20

    oauth_scopes = [
      "https://www.googleapis.com/auth/cloud-platform",
    ]
  }
}

# Grafana for dashboards
resource "kubernetes_namespace" "observability" {
  metadata {
    name = "observability"
  }
  depends_on = [google_container_cluster.observability]
}

resource "helm_release" "grafana" {
  name       = "grafana"
  repository = "https://grafana.github.io/helm-charts"
  chart      = "grafana"
  namespace  = kubernetes_namespace.observability.metadata[0].name

  set {
    name  = "adminPassword"
    value = var.grafana_password
  }

  set {
    name  = "persistence.enabled"
    value = "true"
  }

  set {
    name  = "persistence.size"
    value = "10Gi"
  }
}

# Jaeger for distributed tracing
resource "helm_release" "jaeger" {
  name       = "jaeger"
  repository = "https://jaegertracing.github.io/helm-charts"
  chart      = "jaeger"
  namespace  = kubernetes_namespace.observability.metadata[0].name

  set {
    name  = "storage.type"
    value = "memory"
  }

  set {
    name  = "agent.enabled"
    value = "true"
  }

  set {
    name  = "collector.enabled"
    value = "true"
  }

  set {
    name  = "query.enabled"
    value = "true"
  }
}

variable "grafana_password" {
  description = "Grafana admin password"
  type        = string
  sensitive   = true
  default     = "admin123"
}