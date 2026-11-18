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

variable "project_id" {
  description = "Target Google Cloud project ID (required; never hardcode a real project ID)."
  type        = string

  validation {
    condition     = length(trimspace(var.project_id)) > 0
    error_message = "project_id must be a non-empty Google Cloud project identifier."
  }
}

variable "region" {
  description = "Google Cloud region for Cloud Run and Artifact Registry resources."
  type        = string
  default     = "us-central1"
}

variable "service_name" {
  description = "Name of the Cloud Run v2 service."
  type        = string
  default     = "thrilled-to-announce"
}

variable "artifact_repository_id" {
  description = "Artifact Registry Docker repository identifier."
  type        = string
  default     = "thrilled-repo"
}

variable "container_image" {
  description = "Container image URI to deploy. Defaults to the project Artifact Registry latest tag when empty."
  type        = string
  default     = ""
}

variable "enable_public_invoker_iam_disabled" {
  description = "Disable Cloud Run Invoker IAM checks (--no-invoker-iam-check) for public HTTPS access."
  type        = bool
  default     = true
}

variable "enable_cloud_dlp" {
  description = "Enable Google Cloud Sensitive Data Protection (DLP) inspection in PiiScrubber."
  type        = bool
  default     = true
}

variable "enable_tracing" {
  description = "Enable OpenTelemetry span export to Google Cloud Trace."
  type        = bool
  default     = true
}
