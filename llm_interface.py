import json
import logging
from typing import Generator

import requests

logger = logging.getLogger(__name__)

OPENROUTER_CHAT_URL = "https://openrouter.ai/api/v1/chat/completions"


class LLMInterface:
    def __init__(self, api_key: str, model: str = "google/gemma-4-31b-it:exacto",
                 max_tokens: int = 8192, response_max_tokens: int = 200):
        """Chat-completion LLM backend that talks to a model hosted on OpenRouter.

        Args:
            api_key: OpenRouter API key.
            model: OpenRouter model slug to use for generation.
            max_tokens: Context length budget (informational; OpenRouter enforces
                the model's own context window server-side).
            response_max_tokens: Max tokens to generate per reply.
        """
        if not api_key:
            raise ValueError("OpenRouter API key is required to initialize LLMInterface")

        self.api_key = api_key
        self.model = model
        self.max_tokens = max_tokens
        self.response_max_tokens = response_max_tokens
        self.session = requests.Session()

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
        with self.session.post(OPENROUTER_CHAT_URL, headers=self._headers(), json=payload,
                                stream=True, timeout=60) as response:
            response.raise_for_status()
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
                delta = choices[0].get("delta", {}).get("content")
                if delta:
                    yield delta
