from collections.abc import Iterable

from app.llm.interfaces import GenerationResult, Message, ToolDefinition


def _build_reply(messages: list[Message]) -> str:
    last_user = next((m.content for m in reversed(messages) if m.role == "user"), "")
    return f"[fake-llm] You asked: {last_user!r}. Here's what the retrieved context contained."


class FakeLLMClient:
    """Echoes the prompt it was given, streamed word-by-word. Lets the API/UI/chat
    orchestration be exercised end-to-end without a real provider or API key.
    Swap for a real provider-backed LLMClient when the LLM design lands."""

    def generate(self, messages: list[Message], stream: bool = True) -> Iterable[str]:
        reply = _build_reply(messages)
        words = reply.split(" ")
        for i, word in enumerate(words):
            yield word + (" " if i < len(words) - 1 else "")

    def generate_with_tools(
        self, messages: list[Message], tools: list[ToolDefinition]
    ) -> GenerationResult:
        # Never calls tools -- the fake has no way to decide a tool is worth
        # calling, so the agentic loop (SearchAgent) always terminates in one
        # iteration against this client, same as the pre-tool-calling behavior.
        return GenerationResult(text=_build_reply(messages), tool_calls=None)
