import json
from collections.abc import Iterable

from openai import OpenAI

from app.llm.interfaces import GenerationResult, Message, ToolCall, ToolDefinition


class OpenAIClient:
    """Real LLMClient backed by OpenAI's chat completions API. This is the swap
    point referenced in dependencies.py::get_llm_client() -- to add another
    provider (e.g. Anthropic), implement the same LLMClient protocol
    (generate + generate_with_tools) in a sibling module and add one branch in
    get_llm_client(); nothing else needs to change."""

    def __init__(self, api_key: str, model: str) -> None:
        self._client = OpenAI(api_key=api_key)
        self._model = model

    def generate(self, messages: list[Message], stream: bool = True) -> Iterable[str]:
        payload = [self._to_openai_message(m) for m in messages]
        if not stream:
            response = self._client.chat.completions.create(model=self._model, messages=payload)
            yield response.choices[0].message.content or ""
            return

        response = self._client.chat.completions.create(
            model=self._model, messages=payload, stream=True
        )
        for chunk in response:
            delta = chunk.choices[0].delta.content
            if delta:
                yield delta

    def generate_with_tools(
        self, messages: list[Message], tools: list[ToolDefinition]
    ) -> GenerationResult:
        payload = [self._to_openai_message(m) for m in messages]
        response = self._client.chat.completions.create(
            model=self._model,
            messages=payload,
            tools=[self._to_openai_tool(t) for t in tools] if tools else None,
        )
        message = response.choices[0].message

        if message.tool_calls:
            return GenerationResult(
                tool_calls=[
                    ToolCall(
                        id=tc.id,
                        name=tc.function.name,
                        arguments=json.loads(tc.function.arguments or "{}"),
                    )
                    for tc in message.tool_calls
                ]
            )
        return GenerationResult(text=message.content or "")

    @staticmethod
    def _to_openai_message(message: Message) -> dict:
        out: dict = {"role": message.role, "content": message.content}
        if message.role == "assistant" and message.tool_calls:
            out["tool_calls"] = [
                {
                    "id": tc.id,
                    "type": "function",
                    "function": {"name": tc.name, "arguments": json.dumps(tc.arguments)},
                }
                for tc in message.tool_calls
            ]
        if message.role == "tool":
            out["tool_call_id"] = message.tool_call_id
        return out

    @staticmethod
    def _to_openai_tool(tool: ToolDefinition) -> dict:
        return {
            "type": "function",
            "function": {
                "name": tool.name,
                "description": tool.description,
                "parameters": tool.parameters,
            },
        }
