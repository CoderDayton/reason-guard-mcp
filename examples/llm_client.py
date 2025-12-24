#!/usr/bin/env python3
"""Simple LLM client for testing. Uses env vars."""

from __future__ import annotations

import os

import httpx


class LLMClient:
    """Minimal LLM client - just httpx, no magic."""

    def __init__(
        self,
        base_url: str | None = None,
        model: str | None = None,
        api_key: str | None = None,
    ) -> None:
        self.base_url = base_url or os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1")
        self.model = model or os.environ.get("OPENAI_MODEL", "gpt-4o-mini")
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY", "")

    async def ask(self, prompt: str, system: str | None = None) -> str:
        """Ask LLM a question. Returns response text."""
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        import asyncio

        for attempt in range(3):
            try:
                async with httpx.AsyncClient(timeout=120.0) as client:
                    r = await client.post(
                        f"{self.base_url}/chat/completions",
                        headers={"Authorization": f"Bearer {self.api_key}"},
                        json={
                            "model": self.model,
                            "messages": messages,
                            "max_tokens": 1000,
                            "temperature": 0.1,
                        },
                    )
                    r.raise_for_status()
                    data = r.json()
                    msg = data["choices"][0]["message"]
                    # Some models (like qwen3) put content in 'reasoning' field
                    content = msg.get("content", "") or ""
                    reasoning = msg.get("reasoning", "") or ""
                    # Return content if available, else reasoning, else empty
                    return content if content.strip() else reasoning
            except httpx.HTTPStatusError as e:
                if e.response.status_code == 429 and attempt < 2:
                    await asyncio.sleep(5 * (attempt + 1))  # 5, 10 seconds
                    continue
                raise
        return ""
