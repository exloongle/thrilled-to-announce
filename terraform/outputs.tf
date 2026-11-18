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

output "service_url" {
  description = "Public HTTPS URL of the deployed Thrilled to Announce Cloud Run v2 service."
  value       = google_cloud_run_v2_service.app.uri
}

output "artifact_registry_url" {
  description = "Artifact Registry Docker repository URL for container pushes."
  value       = "${google_artifact_registry_repository.docker_repo.location}-docker.pkg.dev/${var.project_id}/${google_artifact_registry_repository.docker_repo.repository_id}"
}

output "runtime_service_account_email" {
  description = "Email address of the least-privilege Cloud Run runtime service account."
  value       = google_service_account.runtime_sa.email
}

output "secret_id" {
  description = "Secret Manager secret identifier storing the optional Gemini API key."
  value       = google_secret_manager_secret.gemini_api_key.secret_id
}
