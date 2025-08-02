import asyncio
from typing import List

import ollama

from .chat_base import Message, ChatModel


class OllamaChatModel(ChatModel):
    def __init__(
        self,
        model_name="starcoder2:instruct",
        host=None,
    ):
        self.host = host
        self.model_name = model_name

    def generate(self, messages: List[List[Message]], **kwargs) -> List[str]:
        # make sure we have a list of lists
        assert isinstance(messages, list), "messages must be a list of lists"
        assert len(messages) > 0, "messages must have at least one list"
        assert isinstance(messages[0], list), "messages must be a list of lists"

        kwargs.setdefault("temperature", 0.75)
        kwargs.setdefault("top_p", 0.9)
        kwargs.setdefault("num_predict", kwargs.pop("max_tokens", 256))

        return asyncio.run(self._generate_batch(messages, kwargs))

    async def _generate_batch(self, messages, kwargs):
        # Use ollama parallel https://github.com/ollama/ollama/blob/main/docs/faq.md#how-does-ollama-handle-concurrent-requests
        async with asyncio.TaskGroup() as tg:
            out = [tg.create_task(self._generate(message, kwargs)) for message in messages]

        return [i.result() for i in out]

    async def _generate(self, message, kwargs):
        client = ollama.AsyncClient(host=self.host)
        response = await client.chat(
            model=self.model_name,
            messages=message,
            options=kwargs
        )
        return response.message.content.strip()

class OllamaMarkdownChatModel(OllamaChatModel):
    """OllamaChatModel that stops when a Markdown code block is completed"""

    async def _generate(self, message, kwargs):
        client = ollama.AsyncClient(host=self.host)
        response = await client.chat(
            model=self.model_name,
            messages=message,
            options=kwargs,
            stream=True,
        )
        out = []
        last_line = ""

        found_markdown_block = 0
        try:
            async for chunk in response:
                out.append(chunk.message.content)
                last_line += chunk.message.content

                while "\n" in last_line:
                    parts = last_line.split("\n", 1)
                    if "```" in parts[0]:
                        found_markdown_block += 1
                    if found_markdown_block >= 2:
                        raise StopIteration

                    last_line = parts[1]
        except StopIteration:
            # Use StopIteration instead of break to terminate the stream (see ollama-python#210)
            pass

        return "".join(out)
