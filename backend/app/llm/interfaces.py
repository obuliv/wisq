from collections.abc import Iterable
from dataclasses import dataclass
from typing import Literal, Protocol

Role = Literal["system", "user", "assistant"]


@dataclass
class Message:
    role: Role
    content: str


class LLMClient(Protocol):
    """Provider-agnostic chat completion entrypoint. Provider choice, prompt
    template, context-window/history truncation are an LLM deep-dive decision --
    deferred. When stream=True, yields text deltas as they're generated."""

    def generate(self, messages: list[Message], stream: bool = True) -> Iterable[str]: ...
