import logging
import re
from typing import TypeVar

from pydantic import BaseModel, ValidationError

from app.llm.interfaces import LLMClient, Message

logger = logging.getLogger(__name__)

_JSON_BLOCK = re.compile(r"\{.*\}", re.DOTALL)

T = TypeVar("T", bound=BaseModel)


def extract_json(
    llm_client: LLMClient, system_prompt: str, content: str, schema: type[T]
) -> T | None:
    """Calls the LLM with an instruction to return JSON matching `schema`, parses
    and validates the response. Returns None (logging a warning) on any failure --
    enrichment extraction must never raise, since a non-JSON-following LLM (e.g.
    the current FakeLLMClient) is an expected, non-fatal case."""
    messages = [
        Message(role="system", content=system_prompt),
        Message(role="user", content=content),
    ]
    try:
        raw = "".join(llm_client.generate(messages, stream=False))
        match = _JSON_BLOCK.search(raw)
        if not match:
            logger.warning("LLM response contained no JSON object: %r", raw[:200])
            return None
        return schema.model_validate_json(match.group(0))
    except (ValidationError, ValueError) as exc:
        logger.warning("Failed to parse/validate LLM JSON response: %s", exc)
        return None
