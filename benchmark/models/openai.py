import time
import openai
from typing import List

from .chat_base import Message, ChatModel


def openai_edit_prompt_1shot(old: str, instr: str) -> List[Message]:
    return [
        {
            "role": "system",
            "content": """
You are PythonEditGPT. You will be provided the original code snippet and an instruction that specifies the changes you need to make. You will produce the changed code, based on the original code and the instruction given. Only produce the code, do not include any additional prose.
             """.strip(),
        },
        {
            "role": "user",
            "content": """
## Code Before
```py
def add(a, b):
    return a + b
```

## Instruction
Add a "sub" function that subtracts two numbers. Also write docstrings for both functions and change a,b to x,y.
""".strip(),
        },
        {
            "role": "assistant",
            "content": """
## Code After
```py
def add(x, y):
    \"\"\"Adds two numbers.\"\"\"
    return x + y

def sub(x, y):
    \"\"\"Subtracts two numbers.\"\"\"
    return x - y
```
         """.strip(),
        },
        {
            "role": "user",
            "content": f"""
## Code Before
```py
{old}
```
## Instruction
{instr}
""".strip(),
        },
    ]


class OpenAIChatModel(ChatModel):
    def __init__(
        self,
        model_name="gpt-4"
    ):
        import os

        if "ORG_ID" in os.environ:
            openai.organization = os.getenv("ORG_ID")
        assert "OPENAI_API_KEY" in os.environ, "OPENAI_API_KEY must be set"
        openai.api_key = os.getenv("OPENAI_API_KEY")
        self.model_name = model_name
        # Set OPENAI_BASE_URL envar for endpoint

    def generate(self, messages: List[List[Message]], **kwargs) -> List[str]:
        # make sure we have a list of lists
        assert isinstance(messages, list), "messages must be a list of lists"
        assert len(messages) > 0, "messages must have at least one list"
        assert isinstance(
            messages[0], list), "messages must be a list of lists"
        # check that all messages are the same.
        # TODO: support heterogeneous prompts
        assert all(
            m == messages[0] for m in messages), "All prompts must be the same -- batched heterogeneous prompts not supported"
        message = messages[0]

        while True:
            _kwargs = kwargs.copy()
            discard_lengthy = _kwargs.pop("discard_lengthy", False)

            try:
                response = openai.chat.completions.create(
                    model=self.model_name,
                    messages=message,  # type: ignore
                    n=len(messages),
                    stop=_kwargs.pop("stop", None),
                    temperature=_kwargs.pop("temperature", 0.75),
                    top_p=_kwargs.pop("top_p", 0.9),
                    max_tokens=_kwargs.pop("max_tokens", 256),
                    timeout=900.0,
                    **_kwargs,
                )
            except openai.RateLimitError as e:
                print("Rate limit error. Waiting two minutes:", e)
                time.sleep(120)
                continue

            break
        outs = []

        for choice in response.choices:
            if discard_lengthy and choice.finish_reason == "length":
                continue
            text = choice.message.content
            if text is None:
                continue
            outs.append(text.strip())

        return outs
