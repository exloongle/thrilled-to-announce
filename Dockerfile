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

FROM python:3.12-slim

RUN groupadd --gid 10001 appuser && \
    useradd --uid 10001 --gid 10001 --create-home --shell /sbin/nologin appuser && \
    mkdir -p /app /tmp/thrilled_data && \
    chown -R 10001:10001 /app /tmp/thrilled_data

WORKDIR /app

COPY --chown=10001:10001 pyproject.toml README.md LICENSE CONSTITUTION.md ./
COPY --chown=10001:10001 thrilled_to_announce/ ./thrilled_to_announce/
COPY --chown=10001:10001 server/ ./server/
COPY --chown=10001:10001 frontend/ ./frontend/

RUN pip install --no-cache-dir .

ENV PORT=8080 \
    THRILLED_DATA_DIR=/tmp/thrilled_data \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app

USER 10001:10001

EXPOSE 8080

CMD ["uvicorn", "server.main:app", "--host", "0.0.0.0", "--port", "8080"]
