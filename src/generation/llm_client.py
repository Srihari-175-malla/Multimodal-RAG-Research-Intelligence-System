"""
Thin wrapper around the Anthropic Claude API used for all generation tasks
(answering, comparison, claim verification). Swap providers by editing
generate() only — the rest of the codebase depends solely on this interface.
"""
from typing import Optional

import anthropic

from src.utils.config import settings
from src.utils.logging_config import get_logger

logger = get_logger(__name__)


class LLMClient:
    def __init__(self, model: str = None, max_tokens: int = None, temperature: float = None):
        if not settings.anthropic_api_key:
            logger.warning("ANTHROPIC_API_KEY is not set — LLM calls will fail until it is configured.")
        self.client = anthropic.Anthropic(api_key=settings.anthropic_api_key)
        self.model = model or settings.llm_model
        self.max_tokens = max_tokens or settings.llm_max_tokens
        self.temperature = temperature if temperature is not None else settings.llm_temperature

    def generate(self, system_prompt: str, user_prompt: str, max_tokens: Optional[int] = None) -> str:
        response = self.client.messages.create(
            model=self.model,
            max_tokens=max_tokens or self.max_tokens,
            temperature=self.temperature,
            system=system_prompt,
            messages=[{"role": "user", "content": user_prompt}],
        )
        return "".join(block.text for block in response.content if block.type == "text")
