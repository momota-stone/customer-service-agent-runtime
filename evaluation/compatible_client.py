"""Evaluation-only Messages to Chat Completions adapter; no SDK retries."""
import json
from types import SimpleNamespace


def block_dict(block):
    if isinstance(block, dict):
        return block
    if hasattr(block, "model_dump"):
        return block.model_dump()
    return vars(block)


def chat_payload(options):
    messages = []
    if options.get("system"):
        messages.append({"role": "system", "content": options["system"]})
    for message in options["messages"]:
        content = message["content"]
        if isinstance(content, str):
            messages.append({"role": message["role"], "content": content})
            continue
        blocks = [block_dict(block) for block in content]
        if message["role"] == "assistant":
            row = {"role": "assistant", "content": "\n".join(b.get("text", "") for b in blocks if b["type"] == "text") or None}
            calls = [{"id": b["id"], "type": "function", "function": {
                "name": b["name"], "arguments": json.dumps(b["input"], ensure_ascii=False)}}
                for b in blocks if b["type"] == "tool_use"]
            if calls:
                row["tool_calls"] = calls
            messages.append(row)
        else:
            for block in blocks:
                if block["type"] == "tool_result":
                    messages.append({"role": "tool", "tool_call_id": block["tool_use_id"], "content": block["content"]})
                elif block["type"] == "text":
                    messages.append({"role": "user", "content": block["text"]})
                else:
                    raise ValueError("unsupported content block")
    result = {"model": options["model"], "messages": messages,
              "temperature": options.get("temperature", 0), "top_p": 1,
              "max_tokens": options["max_tokens"], "enable_thinking": False}
    if options.get("tools"):
        result["tools"] = [{"type": "function", "function": {
            "name": t["name"], "description": t["description"], "parameters": t["input_schema"]}}
            for t in options["tools"]]
    return result


def message_response(data):
    choice = data["choices"][0]
    message = choice["message"]
    content = []
    if message.get("content"):
        content.append({"type":"text", "text":message["content"]})
    for call in message.get("tool_calls") or []:
        arguments = json.loads(call["function"]["arguments"])
        if not isinstance(arguments, dict):
            raise ValueError("tool arguments must be an object")
        content.append({"type":"tool_use", "id":call["id"], "name":call["function"]["name"], "input":arguments})
    usage = data.get("usage") or {}
    return SimpleNamespace(content=content, stop_reason="max_tokens" if choice.get("finish_reason") == "length" else "end_turn",
                           usage=SimpleNamespace(input_tokens=usage.get("prompt_tokens"), output_tokens=usage.get("completion_tokens")))


class CompatibleClient:
    def __init__(self, sender):
        self.sender = sender
        self.messages = self

    async def create(self, **options):
        return message_response(await self.sender(chat_payload(options)))
