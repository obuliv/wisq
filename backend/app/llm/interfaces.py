from collections.abc import Iterable
from dataclasses import dataclass
from typing import Literal, Protocol

Role = Literal["system", "user", "assistant", "tool"]


@dataclass
class ToolCall:
    id: str
    name: str
    arguments: dict


@dataclass
class Message:
    role: Role
    content: str
    # Set on an assistant message that requested tool calls.
    tool_calls: list[ToolCall] | None = None
    # Set on a role="tool" message: which ToolCall this is the result of.
    tool_call_id: str | None = None


@dataclass
class ToolDefinition:
    name: str
    description: str
    parameters: dict  # JSON Schema object describing the call's arguments


@dataclass
class GenerationResult:
    text: str | None = None
    tool_calls: list[ToolCall] | None = None


class LLMClient(Protocol):
    """Provider-agnostic chat completion entrypoint. Provider choice, prompt
    template, context-window/history truncation are an LLM deep-dive decision --
    deferred. When stream=True, yields text deltas as they're generated."""

    def generate(self, messages: list[Message], stream: bool = True) -> Iterable[str]: ...

    def generate_with_tools(
        self, messages: list[Message], tools: list[ToolDefinition]
    ) -> GenerationResult:
        """Non-streaming: either the model wants to call tools (`tool_calls` set,
        `text` None) or it's done and answering (`text` set, `tool_calls` None).
        Additive alongside `generate` -- extract_json (ingestion/llm_extraction.py)
        and every existing LLM stub/fake only need `generate`, so this is a
        separate protocol method rather than a breaking change to it."""
        ...
