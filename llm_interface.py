import json
import logging
from typing import Generator

import requests

logger = logging.getLogger(__name__)

OPENROUTER_CHAT_URL = "https://openrouter.ai/api/v1/chat/completions"


class LLMInterface:
    def __init__(self, api_key: str, model: str = "google/gemma-4-31b-it:exacto",
                 max_tokens: int = 8192, response_max_tokens: int = 300):
        """Chat-completion LLM backend that talks to a model hosted on OpenRouter.

        Args:
            api_key: OpenRouter API key.
            model: OpenRouter model slug to use for generation.
            max_tokens: Context length budget (informational; OpenRouter enforces
                the model's own context window server-side).
            response_max_tokens: Hard ceiling on tokens generated per reply. This
                is a *backstop*, not the length control - brevity is asked for in
                the system prompt so replies end naturally well below it. It used
                to be 200 and was never overridden by the caller, so ordinary
                replies ran into it and were cut off mid-word.
        """
        if not api_key:
            raise ValueError("OpenRouter API key is required to initialize LLMInterface")

        self.api_key = api_key
        self.model = model
        self.max_tokens = max_tokens
        self.response_max_tokens = response_max_tokens
        self.session = requests.Session()
        # Why the last stream stopped: "stop" (natural end), "length" (hit the
        # cap above, so the trailing sentence is a fragment), or None if the
        # provider never said. Written by generate_response_stream and read by
        # the caller afterwards - callers must hold whatever lock serialises
        # their use of this instance (main.py does, via llm_lock).
        self.last_finish_reason = None

    def _build_messages(self, system_prompt: str, user_message: str, conversation_history: str = ""):
        messages = [{"role": "system", "content": system_prompt}]
        if conversation_history:
            messages.append({"role": "system", "content": f"Conversation so far:\n{conversation_history}"})
        messages.append({"role": "user", "content": user_message})
        return messages

    def _headers(self):
        return {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

    def generate_response(self, system_prompt: str, user_message: str, conversation_history: str = "") -> str:
        """Generate a full (non-streaming) response from the OpenRouter model."""
        payload = {
            "model": self.model,
            "messages": self._build_messages(system_prompt, user_message, conversation_history),
            "temperature": 1.0,
            "top_p": 0.95,
            "max_tokens": self.response_max_tokens,
        }
        response = self.session.post(OPENROUTER_CHAT_URL, headers=self._headers(), json=payload, timeout=60)
        response.raise_for_status()
        data = response.json()
        return data["choices"][0]["message"]["content"].strip()

    def generate_response_stream(self, system_prompt: str, user_message: str,
                                  conversation_history: str = "") -> Generator[str, None, None]:
        """Stream a response from the OpenRouter model, yielding incremental text deltas."""
        payload = {
            "model": self.model,
            "messages": self._build_messages(system_prompt, user_message, conversation_history),
            "temperature": 1.0,
            "top_p": 0.95,
            "max_tokens": self.response_max_tokens,
            "stream": True,
        }
        self.last_finish_reason = None
        with self.session.post(OPENROUTER_CHAT_URL, headers=self._headers(), json=payload,
                                stream=True, timeout=60) as response:
            response.raise_for_status()
            # requests derives the decoding charset from the Content-Type header,
            # and returns ISO-8859-1 for any text/* type that doesn't name one -
            # which is exactly what SSE (text/event-stream) is. Left alone, every
            # multi-byte character the model emits arrives mangled: an em-dash
            # (UTF-8 e2 80 94) becomes "â". That mojibake then
            # flows straight into TTS, and it also defeats the em-dash
            # normalisation in main.py, which no longer has a dash to match.
            response.encoding = "utf-8"
            for line in response.iter_lines(decode_unicode=True):
                if not line or not line.startswith("data:"):
                    continue
                data = line[len("data:"):].strip()
                if data == "[DONE]":
                    return
                try:
                    chunk = json.loads(data)
                except json.JSONDecodeError:
                    continue
                choices = chunk.get("choices") or []
                if not choices:
                    continue
                # Sent on the final chunk of the stream. "length" means the cap
                # truncated the reply, so the caller must not speak the trailing
                # fragment; without this the truncation was completely silent.
                finish_reason = choices[0].get("finish_reason")
                if finish_reason:
                    self.last_finish_reason = finish_reason
                delta = choices[0].get("delta", {}).get("content")
                if delta:
                    yield delta
