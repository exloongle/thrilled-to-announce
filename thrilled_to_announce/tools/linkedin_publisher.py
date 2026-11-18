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
"""Code-enforced Human-in-the-Loop LinkedIn publishing tool and state machine."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any
import uuid

from google.adk.tools import FunctionTool, ToolContext
from google.adk.tools.tool_confirmation import ToolConfirmation

from thrilled_to_announce.memory.style_store import (
    get_default_style_store,
    get_structured_logger,
    scrub_pii_text,
)
from thrilled_to_announce.schemas.contracts import (
    PublishConfirmHttpRequest,
    PublishFeedRequest,
    PublishFeedResponse,
)


def _generate_publish_id() -> str:
    return f"li_post_{uuid.uuid4().hex[:12]}"


def request_publish_approval(
    post_content: str,
    user_id: str = "default_user",
) -> dict[str, Any]:
    """Create a pending Human-in-the-Loop approval record before LinkedIn posting.

    Call this helper when staging a converted LinkedIn announcement for explicit
    user approval in the confirmation modal. It never publishes directly and
    always sets ``status="awaiting_human_confirmation"``.

    Args:
        post_content: Polished LinkedIn announcement text (5 to 3,000 chars) to
            stage for human confirmation.
        user_id: Identifier of the user requesting publication approval.

    Returns:
        Dictionary matching ``PublishFeedResponse`` with
        ``status="awaiting_human_confirmation"``, ``publish_id``, scrubbed
        ``preview_text``, ``message``, and ``recovery_instructions``.
    """
    try:
        scrubbed_post, _, _ = scrub_pii_text((post_content or "").strip())
        req = PublishFeedRequest(
            post_content=scrubbed_post,
            user_id=(user_id or "").strip(),
        )
        publish_id = _generate_publish_id()
        preview_text = req.post_content[:280]
        store = get_default_style_store()
        store.save_pending_publish(
            publish_id=publish_id,
            user_id=req.user_id,
            post_content=req.post_content,
            preview_text=preview_text,
            status="awaiting_human_confirmation",
        )
        get_structured_logger(__name__).info(
            "linkedin_publish_approval_requested",
            publish_id=publish_id,
            user_id=req.user_id,
            human_in_the_loop=True,
        )
        return PublishFeedResponse(
            status="awaiting_human_confirmation",
            publish_id=publish_id,
            preview_text=preview_text,
            message="Approval required: confirm before posting this draft to LinkedIn.",
            recovery_instructions=[
                "Present the confirmation modal and wait for explicit user approval before publishing."
            ],
        ).model_dump()
    except Exception as exc:
        scrubbed_err, _, _ = scrub_pii_text(str(exc))
        error_payload = {"status": "error", "error_message": scrubbed_err}
        get_structured_logger(__name__).warning(
            "tool_recoverable_error",
            tool="request_publish_approval",
            **error_payload,
        )
        return PublishFeedResponse(
            status="recoverable_error",
            publish_id="li_post_invalid",
            preview_text="",
            message=f"Cannot stage LinkedIn post ({scrubbed_err}).",
            recovery_instructions=[
                "Please provide post_content between 5 and 3000 characters and try again."
            ],
        ).model_dump()


def confirm_publish_decision(
    publish_id: str,
    confirmed: bool,
) -> dict[str, Any]:
    """Finalize a pending Human-in-the-Loop LinkedIn publish decision.

    Call this function after the human user clicks ``Post`` (``confirmed=True``)
    or ``Cancel`` (``confirmed=False``) on the ``Post to LinkedIn?`` modal.

    Args:
        publish_id: Pending ``li_post_*`` identifier returned by
            ``request_publish_approval`` or ``publish_to_linkedin_feed``.
        confirmed: Explicit boolean decision from the human reviewer
            (``True`` to publish, ``False`` to cancel).

    Returns:
        Dictionary matching ``PublishFeedResponse`` with ``status="published"``
        when ``confirmed=True``, ``status="cancelled_by_human"`` when
        ``confirmed=False``, or ``status="recoverable_error"`` if ``publish_id``
        is unknown.
    """
    try:
        if not isinstance(confirmed, bool):
            raise ValueError(
                "Parameter 'confirmed' must be a strict boolean (True or False)."
            )
        if not isinstance(publish_id, str):
            raise ValueError("Parameter 'publish_id' must be a string.")
        req = PublishConfirmHttpRequest(
            publish_id=publish_id.strip(),
            confirmed=confirmed,
        )
        store = get_default_style_store()
        existing = store.get_pending_publish(req.publish_id)
        if existing is None:
            return PublishFeedResponse(
                status="recoverable_error",
                publish_id=req.publish_id,
                preview_text="",
                message=f"Unknown or expired publish_id '{req.publish_id}'.",
                recovery_instructions=[
                    "Please call request_publish_approval to stage a fresh confirmation request and try again."
                ],
            ).model_dump()

        existing_status = str(existing.get("status", ""))
        if existing_status != "awaiting_human_confirmation":
            return PublishFeedResponse(
                status="recoverable_error",
                publish_id=req.publish_id,
                preview_text=str(existing.get("preview_text", "")),
                message=(
                    f"Publish request '{req.publish_id}' is already finalized "
                    f"with terminal status '{existing_status}'."
                ),
                recovery_instructions=[
                    "Please call request_publish_approval to stage a fresh confirmation request and try again."
                ],
            ).model_dump()

        if req.confirmed:
            updated = store.save_pending_publish(
                publish_id=req.publish_id,
                user_id=str(existing["user_id"]),
                post_content=str(existing["post_content"]),
                preview_text=str(existing["preview_text"]),
                status="published",
            )
            get_structured_logger(__name__).info(
                "linkedin_post_published",
                publish_id=req.publish_id,
                user_id=updated["user_id"],
            )
            return PublishFeedResponse(
                status="published",
                publish_id=req.publish_id,
                preview_text=str(updated["preview_text"]),
                message="Confirmed by user and published to LinkedIn feed.",
                recovery_instructions=[],
            ).model_dump()

        cancelled = store.save_pending_publish(
            publish_id=req.publish_id,
            user_id=str(existing["user_id"]),
            post_content=str(existing["post_content"]),
            preview_text=str(existing["preview_text"]),
            status="cancelled_by_human",
        )
        get_structured_logger(__name__).info(
            "linkedin_post_cancelled_by_human",
            publish_id=req.publish_id,
            user_id=cancelled["user_id"],
        )
        return PublishFeedResponse(
            status="cancelled_by_human",
            publish_id=req.publish_id,
            preview_text=str(cancelled["preview_text"]),
            message="Publication cancelled by user; draft kept in chat.",
            recovery_instructions=[],
        ).model_dump()
    except Exception as exc:
        scrubbed_err, _, _ = scrub_pii_text(str(exc))
        error_payload = {"status": "error", "error_message": scrubbed_err}
        get_structured_logger(__name__).warning(
            "tool_recoverable_error",
            tool="confirm_publish_decision",
            **error_payload,
        )
        safe_pid = (
            publish_id.strip()[:128]
            if isinstance(publish_id, str) and len(publish_id.strip()) >= 3
            else "li_post_invalid"
        )
        return PublishFeedResponse(
            status="recoverable_error",
            publish_id=safe_pid,
            preview_text="",
            message=f"Invalid confirmation payload ({scrubbed_err}).",
            recovery_instructions=[
                "Please provide a valid publish_id (3-128 characters) and boolean confirmed flag and try again."
            ],
        ).model_dump()


def publish_to_linkedin_feed(
    post_content: str,
    user_id: str = "default_user",
    tool_context: ToolContext | None = None,
) -> dict[str, Any]:
    """Publish a converted LinkedIn post strictly after Human-in-the-Loop approval.

    Call this tool ONLY when the user explicitly asks to publish a polished
    LinkedIn announcement. Wrapped in ``FunctionTool(..., require_confirmation=True)``
    and guarded internally via ``ToolConfirmation``, it halts with
    ``status="awaiting_human_confirmation"`` whenever human confirmation has not
    yet been granted, cancels with ``status="cancelled_by_human"`` if rejected,
    and publishes ONLY when ``tool_context.tool_confirmation.confirmed`` is
    ``True``.

    Args:
        post_content: Final polished LinkedIn announcement text (5 to 3,000
            characters) to publish after confirmation.
        user_id: Identifier of the user whose LinkedIn feed receives the post.
        tool_context: ADK ``ToolContext`` carrying ``ToolConfirmation`` state
            and ``request_confirmation`` callback hooks.

    Returns:
        Dictionary matching ``PublishFeedResponse`` with ``status``,
        ``publish_id``, ``preview_text``, ``message``, and
        ``recovery_instructions``.
    """
    try:
        if not isinstance(post_content, str) or not isinstance(user_id, str):
            raise ValueError("post_content and user_id must be strings.")
        scrubbed_post, _, _ = scrub_pii_text(post_content.strip())
        req = PublishFeedRequest(
            post_content=scrubbed_post,
            user_id=user_id.strip(),
        )

        confirmation_obj: ToolConfirmation | Any | None = (
            getattr(tool_context, "tool_confirmation", None)
            if tool_context is not None
            else None
        )

        if confirmation_obj is None:
            pending_dict = request_publish_approval(
                post_content=req.post_content,
                user_id=req.user_id,
            )
            if (
                tool_context is not None
                and hasattr(tool_context, "request_confirmation")
                and getattr(tool_context, "function_call_id", None)
            ):
                tool_context.request_confirmation(
                    hint="Confirm before publishing this announcement to your LinkedIn feed.",
                    payload={
                        "publish_id": pending_dict["publish_id"],
                        "preview_text": pending_dict["preview_text"],
                    },
                )
                actions = getattr(tool_context, "actions", None)
                if actions is not None and hasattr(actions, "skip_summarization"):
                    actions.skip_summarization = True
            return pending_dict

        raw_confirmed = getattr(confirmation_obj, "confirmed", None)
        if raw_confirmed is not True and raw_confirmed is not False:
            return PublishFeedResponse(
                status="recoverable_error",
                publish_id="li_post_invalid",
                preview_text="",
                message="ToolConfirmation.confirmed must be a strict boolean (True or False).",
                recovery_instructions=[
                    "Please supply an explicit boolean confirmed flag in ToolConfirmation and try again."
                ],
            ).model_dump()

        payload = getattr(confirmation_obj, "payload", None)
        if isinstance(payload, Mapping) and "publish_id" in payload:
            raw_pid = payload["publish_id"]
            if not isinstance(raw_pid, str) or not raw_pid.strip():
                return PublishFeedResponse(
                    status="recoverable_error",
                    publish_id="li_post_invalid",
                    preview_text="",
                    message="ToolConfirmation payload contains an empty or non-string publish_id.",
                    recovery_instructions=[
                        "Please provide a valid non-empty string publish_id (3-128 characters) and try again."
                    ],
                ).model_dump()
            return confirm_publish_decision(raw_pid.strip(), confirmed=raw_confirmed)

        if raw_confirmed is False:
            publish_id = _generate_publish_id()
            preview_text = req.post_content[:280]
            get_default_style_store().save_pending_publish(
                publish_id=publish_id,
                user_id=req.user_id,
                post_content=req.post_content,
                preview_text=preview_text,
                status="cancelled_by_human",
            )
            return PublishFeedResponse(
                status="cancelled_by_human",
                publish_id=publish_id,
                preview_text=preview_text,
                message="Publication rejected by human reviewer.",
                recovery_instructions=[],
            ).model_dump()

        publish_id = _generate_publish_id()
        preview_text = req.post_content[:280]
        get_default_style_store().save_pending_publish(
            publish_id=publish_id,
            user_id=req.user_id,
            post_content=req.post_content,
            preview_text=preview_text,
            status="published",
        )
        get_structured_logger(__name__).info(
            "linkedin_post_published_via_adk_confirmation",
            publish_id=publish_id,
            user_id=req.user_id,
        )
        return PublishFeedResponse(
            status="published",
            publish_id=publish_id,
            preview_text=preview_text,
            message="Confirmed via ADK ToolConfirmation and published to LinkedIn feed.",
            recovery_instructions=[],
        ).model_dump()
    except Exception as exc:
        scrubbed_err, _, _ = scrub_pii_text(str(exc))
        error_payload = {"status": "error", "error_message": scrubbed_err}
        get_structured_logger(__name__).warning(
            "tool_recoverable_error",
            tool="publish_to_linkedin_feed",
            **error_payload,
        )
        return PublishFeedResponse(
            status="recoverable_error",
            publish_id="li_post_invalid",
            preview_text="",
            message=f"LinkedIn publish validation failed ({scrubbed_err}).",
            recovery_instructions=[
                "Please provide post_content between 5 and 3000 characters and try again."
            ],
        ).model_dump()


publish_to_linkedin_tool = FunctionTool(
    publish_to_linkedin_feed,
    require_confirmation=True,
)
