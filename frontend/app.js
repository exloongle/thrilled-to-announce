/*
 * Copyright 2026 Google LLC
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     https://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

(() => {
  "use strict";

  const USER_ID_STORAGE_KEY = "ta_persistent_user_id";

  const FALLBACK_WELCOME = {
    greeting:
      "Hi! Send me a blunt thought and I'll turn it into a polished LinkedIn announcement—or paste a LinkedIn post and I'll tell you what it really means.",
    starter_prompts: [
      "I am tired of my current company and just wanted to find something not that boring",
      "Thrilled to announce that after 4 impactful years driving synergistic transformation, I'm stepping back to explore fractional advisory horizons!",
    ],
  };

  const chatViewport = document.getElementById("chatViewport");
  const chatStream = document.getElementById("chatStream");
  const composerForm = document.getElementById("composerForm");
  const messageInput = document.getElementById("messageInput");
  const sendBtn = document.getElementById("sendBtn");
  const newChatBtn = document.getElementById("newChatBtn");

  const hitlOverlay = document.getElementById("hitlOverlay");
  const hitlPreview = document.getElementById("hitlPreview");
  const hitlCancelBtn = document.getElementById("hitlCancelBtn");
  const hitlConfirmBtn = document.getElementById("hitlConfirmBtn");

  let currentSessionId = null;
  let isSending = false;
  let pendingPublishId = null;
  let lastLinkedinDraft = "";
  let cachedWelcome = null;

  function getOrCreateUserId() {
    try {
      const existing = window.localStorage.getItem(USER_ID_STORAGE_KEY);
      if (existing && existing.trim().length > 0) {
        return existing.trim();
      }
      const generated =
        typeof crypto !== "undefined" && typeof crypto.randomUUID === "function"
          ? `user_${crypto.randomUUID().replace(/-/g, "").slice(0, 16)}`
          : `user_${Math.random().toString(36).slice(2, 14)}${Date.now().toString(36)}`;
      window.localStorage.setItem(USER_ID_STORAGE_KEY, generated);
      return generated;
    } catch (_err) {
      return "default_user";
    }
  }

  const persistentUserId = getOrCreateUserId();

  function createAvatar() {
    const avatar = document.createElement("div");
    avatar.className = "ta-avatar";
    avatar.setAttribute("aria-hidden", "true");
    avatar.textContent = "ta";
    return avatar;
  }

  function scrollToBottom() {
    window.requestAnimationFrame(() => {
      chatViewport.scrollTop = chatViewport.scrollHeight;
    });
  }

  function updateSendButtonState() {
    const hasValidLength = messageInput.value.trim().length >= 2;
    sendBtn.disabled = isSending || !hasValidLength;
  }

  function formatChipLabel(promptText) {
    const trimmed = promptText.trim();
    const unquoted = trimmed.replace(/^["“]+|["”]+$/g, "");
    const maxLen = 48;
    const display =
      unquoted.length > maxLen ? `${unquoted.slice(0, maxLen).trimEnd()}...` : unquoted;
    return `"${display}"`;
  }

  function stripOuterQuotes(promptText) {
    return promptText.trim().replace(/^["“]+|["”]+$/g, "");
  }

  function renderWelcome(welcomeData) {
    const row = document.createElement("div");
    row.className = "msg-row-assistant";
    row.appendChild(createAvatar());

    const col = document.createElement("div");
    col.className = "welcome-col";

    const bubble = document.createElement("div");
    bubble.className = "assistant-card";
    const text = document.createElement("p");
    text.className = "reply-text";
    text.textContent = welcomeData.greeting || FALLBACK_WELCOME.greeting;
    bubble.appendChild(text);
    col.appendChild(bubble);

    const prompts = Array.isArray(welcomeData.starter_prompts)
      ? welcomeData.starter_prompts
      : [];
    if (prompts.length > 0) {
      const chipsWrap = document.createElement("div");
      chipsWrap.className = "starter-chips";

      prompts.forEach((rawPrompt) => {
        const cleanPrompt = stripOuterQuotes(String(rawPrompt || ""));
        if (cleanPrompt.length < 2) {
          return;
        }
        const chip = document.createElement("button");
        chip.type = "button";
        chip.className = "starter-chip";
        chip.textContent = formatChipLabel(cleanPrompt);
        chip.title = cleanPrompt;
        chip.addEventListener("click", () => {
          if (!isSending) {
            sendMessage(cleanPrompt);
          }
        });
        chipsWrap.appendChild(chip);
      });

      if (chipsWrap.childElementCount > 0) {
        col.appendChild(chipsWrap);
      }
    }

    row.appendChild(col);
    chatStream.appendChild(row);
    scrollToBottom();
  }

  async function loadWelcome() {
    if (cachedWelcome) {
      renderWelcome(cachedWelcome);
      return;
    }
    try {
      const response = await fetch("/api/welcome", {
        method: "GET",
        headers: { Accept: "application/json" },
      });
      if (!response.ok) {
        throw new Error(`HTTP ${response.status}`);
      }
      const data = await response.json();
      cachedWelcome = data;
      renderWelcome(data);
    } catch (_err) {
      renderWelcome(FALLBACK_WELCOME);
    }
  }

  function appendUserBubble(text) {
    const row = document.createElement("div");
    row.className = "msg-row-user";

    const bubble = document.createElement("div");
    bubble.className = "user-bubble";
    bubble.textContent = text;

    row.appendChild(bubble);
    chatStream.appendChild(row);
    scrollToBottom();
  }

  function appendThinkingBubble() {
    const row = document.createElement("div");
    row.className = "msg-row-assistant";
    row.setAttribute("role", "status");
    row.setAttribute("aria-label", "Thinking");
    row.appendChild(createAvatar());

    const bubble = document.createElement("div");
    bubble.className = "thinking-bubble";
    for (let i = 0; i < 3; i += 1) {
      const dot = document.createElement("span");
      dot.className = "thinking-dot";
      bubble.appendChild(dot);
    }

    row.appendChild(bubble);
    chatStream.appendChild(row);
    scrollToBottom();
    return row;
  }

  function appendStatusBubble(messageText, isPublished) {
    const row = document.createElement("div");
    row.className = "msg-row-assistant";
    row.appendChild(createAvatar());

    const bubble = document.createElement("div");
    bubble.className = `publish-status-bubble${isPublished ? " is-published" : ""}`;
    bubble.textContent = messageText;

    row.appendChild(bubble);
    chatStream.appendChild(row);
    scrollToBottom();
  }

  async function copyTextToClipboard(text, buttonEl) {
    try {
      if (navigator.clipboard && typeof navigator.clipboard.writeText === "function") {
        await navigator.clipboard.writeText(text);
      } else {
        const temp = document.createElement("textarea");
        temp.value = text;
        temp.style.position = "fixed";
        temp.style.opacity = "0";
        document.body.appendChild(temp);
        temp.select();
        document.execCommand("copy");
        document.body.removeChild(temp);
      }
      buttonEl.textContent = "Copied ✓";
      buttonEl.classList.add("is-copied");
      window.setTimeout(() => {
        buttonEl.textContent = "Copy";
        buttonEl.classList.remove("is-copied");
      }, 1500);
    } catch (_err) {
      buttonEl.textContent = "Copy";
    }
  }

  function appendAssistantReply(res) {
    const row = document.createElement("div");
    row.className = "msg-row-assistant";
    row.appendChild(createAvatar());

    const card = document.createElement("div");
    card.className = `assistant-card${res.guardrail_blocked ? " is-guardrail" : ""}`;

    const metaRow = document.createElement("div");
    metaRow.className = "card-meta-row";

    const labelEl = document.createElement("span");
    const labelText =
      res.label || (res.guardrail_blocked ? "Confidentiality Check" : "Assistant");
    const isLinkedinLabel =
      !res.guardrail_blocked &&
      (Boolean(res.can_post_to_linkedin) ||
        labelText.toLowerCase().includes("linkedin"));

    labelEl.className = "card-label";
    if (res.guardrail_blocked) {
      labelEl.classList.add("is-guardrail");
    } else if (isLinkedinLabel) {
      labelEl.classList.add("is-linkedin");
    }
    labelEl.textContent = labelText;
    metaRow.appendChild(labelEl);

    if (res.preference_saved) {
      const memNote = document.createElement("span");
      memNote.className = "memory-note is-saved";
      memNote.textContent = "✓ Style preference saved";
      metaRow.appendChild(memNote);
    } else if (res.used_saved_preference) {
      const memNote = document.createElement("span");
      memNote.className = "memory-note";
      memNote.textContent = "Using your saved style";
      metaRow.appendChild(memNote);
    }

    card.appendChild(metaRow);

    const replyText = String(res.reply_text || "");
    const bodyEl = document.createElement("p");
    bodyEl.className = "reply-text";
    bodyEl.textContent = replyText;
    card.appendChild(bodyEl);

    if (res.can_post_to_linkedin && replyText.trim().length >= 5) {
      lastLinkedinDraft = replyText;
    }

    const showCopy =
      !res.guardrail_blocked &&
      replyText.trim().length > 0 &&
      (Boolean(res.can_post_to_linkedin) || res.detected_style !== "PUBLISH_REQUEST");

    if (showCopy) {
      const actionsRow = document.createElement("div");
      actionsRow.className = "card-actions";

      const copyBtn = document.createElement("button");
      copyBtn.type = "button";
      copyBtn.className = "action-link";
      copyBtn.textContent = "Copy";
      copyBtn.addEventListener("click", () => {
        copyTextToClipboard(replyText, copyBtn);
      });
      actionsRow.appendChild(copyBtn);

      if (res.can_post_to_linkedin) {
        const sep = document.createElement("span");
        sep.className = "action-sep";
        sep.textContent = "·";
        actionsRow.appendChild(sep);

        const postBtn = document.createElement("button");
        postBtn.type = "button";
        postBtn.className = "action-link";
        postBtn.textContent = "Post to LinkedIn";
        postBtn.addEventListener("click", () => {
          requestPublishApproval(replyText);
        });
        actionsRow.appendChild(postBtn);
      }

      card.appendChild(actionsRow);
    }

    row.appendChild(card);
    chatStream.appendChild(row);
    scrollToBottom();
  }

  function openHitlModal(publishId, previewContent) {
    pendingPublishId = publishId;
    const cleanPreview = String(previewContent || "").trim();
    hitlPreview.textContent = cleanPreview ? `"${cleanPreview}"` : "";
    hitlCancelBtn.disabled = false;
    hitlConfirmBtn.disabled = false;
    hitlOverlay.classList.remove("hidden");
    hitlConfirmBtn.focus();
  }

  function closeHitlModal() {
    hitlOverlay.classList.add("hidden");
    pendingPublishId = null;
    hitlCancelBtn.disabled = false;
    hitlConfirmBtn.disabled = false;
  }

  async function requestPublishApproval(postContent) {
    const contentToPublish = String(postContent || lastLinkedinDraft || "").trim();
    if (contentToPublish.length < 5) {
      appendStatusBubble("Draft is too short to publish to LinkedIn.", false);
      return;
    }

    try {
      const response = await fetch("/api/publish/request", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
          Accept: "application/json",
        },
        body: JSON.stringify({
          post_content: contentToPublish,
          user_id: persistentUserId,
        }),
      });

      if (!response.ok) {
        throw new Error(`HTTP ${response.status}`);
      }

      const data = await response.json();
      if (data.status === "recoverable_error" || !data.publish_id) {
        appendStatusBubble(
          data.message || "Unable to prepare LinkedIn post right now.",
          false
        );
        return;
      }

      openHitlModal(data.publish_id, data.preview_text || contentToPublish);
    } catch (_err) {
      appendStatusBubble("Could not reach server to prepare LinkedIn post.", false);
    }
  }

  async function submitHitlDecision(confirmed) {
    if (!pendingPublishId) {
      closeHitlModal();
      return;
    }

    const publishId = pendingPublishId;
    hitlCancelBtn.disabled = true;
    hitlConfirmBtn.disabled = true;

    try {
      const response = await fetch("/api/publish/confirm", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
          Accept: "application/json",
        },
        body: JSON.stringify({
          publish_id: publishId,
          confirmed: Boolean(confirmed),
        }),
      });

      if (!response.ok) {
        throw new Error(`HTTP ${response.status}`);
      }

      const data = await response.json();
      closeHitlModal();

      if (confirmed && data.status === "published") {
        appendStatusBubble("✓ Posted to LinkedIn", true);
      } else if (!confirmed || data.status === "cancelled_by_human") {
        appendStatusBubble("Post cancelled", false);
      } else {
        appendStatusBubble(data.message || "Post cancelled", false);
      }
    } catch (_err) {
      closeHitlModal();
      appendStatusBubble(
        confirmed ? "Could not confirm LinkedIn post." : "Post cancelled",
        false
      );
    } finally {
      messageInput.focus();
    }
  }

  async function sendMessage(rawText) {
    const trimmed = String(rawText || "").trim();
    if (isSending || trimmed.length < 2) {
      return;
    }

    isSending = true;
    messageInput.value = "";
    updateSendButtonState();

    appendUserBubble(trimmed);
    const thinkingRow = appendThinkingBubble();

    const payload = {
      message: trimmed,
      user_id: persistentUserId,
      session_id: currentSessionId,
    };

    try {
      const response = await fetch("/api/chat", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
          Accept: "application/json",
        },
        body: JSON.stringify(payload),
      });

      if (!response.ok) {
        throw new Error(`HTTP ${response.status}`);
      }

      const res = await response.json();
      thinkingRow.remove();

      if (res.session_id) {
        currentSessionId = res.session_id;
      }

      appendAssistantReply(res);

      if (res.hitl_required) {
        if (res.pending_publish_id) {
          openHitlModal(
            res.pending_publish_id,
            lastLinkedinDraft || res.reply_text || ""
          );
        } else {
          await requestPublishApproval(lastLinkedinDraft || res.reply_text || "");
        }
      }
    } catch (_err) {
      thinkingRow.remove();
      appendAssistantReply({
        label: "Connection Issue",
        reply_text: "Something went wrong reaching the server. Please try again.",
        guardrail_blocked: true,
        can_post_to_linkedin: false,
      });
    } finally {
      isSending = false;
      updateSendButtonState();
      if (hitlOverlay.classList.contains("hidden")) {
        messageInput.focus();
      }
    }
  }

  function resetChatSession() {
    if (isSending) {
      return;
    }
    currentSessionId = null;
    lastLinkedinDraft = "";
    closeHitlModal();
    chatStream.innerHTML = "";
    loadWelcome();
    messageInput.value = "";
    updateSendButtonState();
    messageInput.focus();
  }

  messageInput.addEventListener("input", updateSendButtonState);

  composerForm.addEventListener("submit", (event) => {
    event.preventDefault();
    sendMessage(messageInput.value);
  });

  newChatBtn.addEventListener("click", resetChatSession);

  hitlCancelBtn.addEventListener("click", () => {
    submitHitlDecision(false);
  });

  hitlConfirmBtn.addEventListener("click", () => {
    submitHitlDecision(true);
  });

  document.addEventListener("keydown", (event) => {
    if (event.key === "Escape" && !hitlOverlay.classList.contains("hidden")) {
      submitHitlDecision(false);
    }
  });

  loadWelcome();
  updateSendButtonState();
})();
