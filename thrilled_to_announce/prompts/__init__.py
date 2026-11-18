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
"""Constitutional system instructions package."""

from thrilled_to_announce.prompts.constitution import (
    COORDINATOR_INSTRUCTION,
    GLOBAL_CONSTITUTION,
    REACTION_SIMULATOR_INSTRUCTION,
    SELF_EVAL_AUDITOR_INSTRUCTION,
    SUBTEXT_ANALYST_INSTRUCTION,
    TRANSLATOR_INSTRUCTION,
)

__all__ = [
    "COORDINATOR_INSTRUCTION",
    "GLOBAL_CONSTITUTION",
    "REACTION_SIMULATOR_INSTRUCTION",
    "SELF_EVAL_AUDITOR_INSTRUCTION",
    "SUBTEXT_ANALYST_INSTRUCTION",
    "TRANSLATOR_INSTRUCTION",
]
