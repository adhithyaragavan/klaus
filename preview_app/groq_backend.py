"""LLMBackend-shaped client for Groq's free-tier, OpenAI-compatible endpoint.

This is intentionally kept outside src/ — it is not one of Klaus's core, on-prem backends
(ROCmVLLMBackend / LocalDevBackend). It exists only to power the free-hosted Streamlit preview,
since Streamlit Community Cloud has no GPU and can't run an LLM itself. See README.md's "Live
Demo" section and docs/AMD_VERIFICATION.md for the real, AMD-verified production backend.
"""
from __future__ import annotations

import requests

GROQ_BASE_URL = "https://api.groq.com/openai/v1"
DEFAULT_MODEL = "llama-3.3-70b-versatile"


class GroqPreviewBackend:
    def __init__(self, api_key: str, model: str = DEFAULT_MODEL) -> None:
        self.api_key = api_key
        self.model = model

    def generate(self, prompt: str) -> str:
        response = requests.post(
            f"{GROQ_BASE_URL}/chat/completions",
            headers={"Authorization": f"Bearer {self.api_key}"},
            json={
                "model": self.model,
                "messages": [{"role": "user", "content": prompt}],
                "temperature": 0.0,
            },
            timeout=30,
        )
        response.raise_for_status()
        return response.json()["choices"][0]["message"]["content"]
