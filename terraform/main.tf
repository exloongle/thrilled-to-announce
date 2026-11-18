# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

locals {
  required_apis = toset([
    "aiplatform.googleapis.com",
    "artifactregistry.googleapis.com",
    "cloudbuild.googleapis.com",
    "cloudtrace.googleapis.com",
    "dlp.googleapis.com",
    "run.googleapis.com",
    "secretmanager.googleapis.com",
  ])

  least_privilege_roles = toset([
    "roles/aiplatform.user",
    "roles/cloudtrace.agent",
    "roles/dlp.user",
    "roles/logging.logWriter",
  ])

  resolved_container_image = (
    var.container_image != ""
    ? var.container_image
    : "${var.region}-docker.pkg.dev/${var.project_id}/${google_artifact_registry_repository.docker_repo.repository_id}/${var.service_name}:latest"
  )
}

resource "google_project_service" "required_apis" {
  for_each = local.required_apis

  project            = var.project_id
  service            = each.value
  disable_on_destroy = false
}

resource "google_artifact_registry_repository" "docker_repo" {
  project       = var.project_id
  location      = var.region
  repository_id = var.artifact_repository_id
  description   = "Container images for Thrilled to Announce"
  format        = "DOCKER"

  depends_on = [
    google_project_service.required_apis,
  ]
}

resource "google_service_account" "runtime_sa" {
  project      = var.project_id
  account_id   = "thrilled-agent-sa"
  display_name = "Thrilled to Announce Runtime Service Account"
  description  = "Least-privilege Cloud Run runtime identity for Thrilled to Announce"

  depends_on = [
    google_project_service.required_apis,
  ]
}

resource "google_project_iam_member" "runtime_sa_roles" {
  for_each = local.least_privilege_roles

  project = var.project_id
  role    = each.value
  member  = "serviceAccount:${google_service_account.runtime_sa.email}"
}

resource "google_secret_manager_secret" "gemini_api_key" {
  project   = var.project_id
  secret_id = "gemini-api-key"

  replication {
    auto {}
  }

  depends_on = [
    google_project_service.required_apis,
  ]
}

# Seeds an initial Vertex-ADC sentinel version so Cloud Run secret_key_ref
# version = "latest" succeeds on first apply while ignore_changes preserves
# out-of-band secret rotation.
resource "google_secret_manager_secret_version" "gemini_api_key_initial" {
  secret      = google_secret_manager_secret.gemini_api_key.id
  secret_data = "none"

  lifecycle {
    ignore_changes = [secret_data]
  }
}

resource "google_secret_manager_secret_iam_member" "runtime_secret_accessor" {
  project   = var.project_id
  secret_id = google_secret_manager_secret.gemini_api_key.id
  role      = "roles/secretmanager.secretAccessor"
  member    = "serviceAccount:${google_service_account.runtime_sa.email}"
}

resource "google_cloud_run_v2_service" "app" {
  name                 = var.service_name
  project              = var.project_id
  location             = var.region
  ingress              = "INGRESS_TRAFFIC_ALL"
  invoker_iam_disabled = var.enable_public_invoker_iam_disabled
  deletion_protection  = false

  template {
    service_account = google_service_account.runtime_sa.email

    scaling {
      min_instance_count = 0
      max_instance_count = 5
    }

    containers {
      image = local.resolved_container_image

      ports {
        container_port = 8080
      }

      startup_probe {
        initial_delay_seconds = 2
        period_seconds        = 10
        timeout_seconds       = 5
        failure_threshold     = 3

        http_get {
          path = "/health"
          port = 8080
        }
      }

      liveness_probe {
        initial_delay_seconds = 10
        period_seconds        = 30
        timeout_seconds       = 5
        failure_threshold     = 3

        http_get {
          path = "/health"
          port = 8080
        }
      }

      env {
        name  = "GOOGLE_GENAI_USE_VERTEXAI"
        value = "TRUE"
      }

      env {
        name  = "GOOGLE_CLOUD_PROJECT"
        value = var.project_id
      }

      env {
        name  = "GOOGLE_CLOUD_LOCATION"
        value = var.region
      }

      env {
        name  = "GEMINI_API_KEY_SECRET_ID"
        value = google_secret_manager_secret.gemini_api_key.secret_id
      }

      env {
        name  = "ENABLE_CLOUD_DLP"
        value = var.enable_cloud_dlp ? "TRUE" : "FALSE"
      }

      env {
        name  = "OTEL_EXPORT_TO_CLOUD"
        value = var.enable_tracing ? "TRUE" : "FALSE"
      }

      env {
        name  = "THRILLED_DATA_DIR"
        value = "/tmp/thrilled_data"
      }

      env {
        name  = "THRILLED_FLASH_MODEL"
        value = "gemini-4-hydrogen"
      }

      env {
        name  = "THRILLED_PRO_MODEL"
        value = "gemini-4-hydrogen"
      }

      env {
        name = "GEMINI_API_KEY"

        value_source {
          secret_key_ref {
            secret  = google_secret_manager_secret.gemini_api_key.secret_id
            version = "latest"
          }
        }
      }
    }
  }

  depends_on = [
    google_project_service.required_apis,
    google_project_iam_member.runtime_sa_roles,
    google_secret_manager_secret_iam_member.runtime_secret_accessor,
    google_secret_manager_secret_version.gemini_api_key_initial,
  ]
}
