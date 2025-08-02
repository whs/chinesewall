import io
import json
from hashlib import blake2b
from pathlib import Path
from typing import List, Callable, Tuple, Optional

from .chat_base import ChatModel, Message
from .edit_base import EditModel, EditCommand, EditResponse, python_markdown_codeblock_extract

ArchitectPromptFunction = Callable[[EditCommand], List[Message]]
EditorPromptFunction = Callable[[str, str], List[Message]]

def architect_prompt_1shot(cmd: EditCommand) -> List[Message]:
    return [
        {"role": "system", "content": f"""You are PythonTeacherGPT. You will be provided the original code snippet and an instruction that specifies the changes to complete the task. You must not complete the task. Instead, you will add comments to the code, and NEVER create, modify or remove any code. Each comments is a detailed instructions that a novice programmer could follow to complete the task without reading the original instruction or understanding other concepts in other fields than computer science, such as mathematics or electronics.
        
Your comments must be placed at the spot where you would normally make such edit. You may make multiple comments. Your comment must starts with `EDIT:`. Only produce the commented code, do not include any additional prose."""},
        {"role": "user", "content": f"""# Instructions
Return the largest between the two numbers

# Code
```py
# Max function
def max(a, b):
    return 0
```"""},
        {"role": "assistant", "content": f"""```py
# Max function
# EDIT: Add type hint of a, b and return value which are all `int`
def max(a, b):
    # EDIT: Implement a function that return the largest of `a` or `b`
    # 1. If `a` is greater than `b`, return `a`
    # 2. If `b` is greater than `a`, return `b`
    # Otherwise a and b is equal, and any can be returned
    return 0
```"""},
        {"role": "user", "content": f"""# Instructions
{cmd["instruction"]}

# Code
```py
{cmd["content"]}
```"""},
    ]

def editor_prompt_with_instr_1shot(instruction: str, annotated_code: str) -> List[Message]:
    return [
        {"role": "user", "content": f"""
Update this code by following the instruction in all comments that starts with `EDIT: `. Then remove the EDIT comments. If there is conflict between problem statement and EDIT comments, follow the comment.
Only produce the code, do not include any additional prose.

## Problem
>Return the smallest between the two numbers

## Code
```py
def sum(a: int, b: int) -> int:
    # Return a sum of a +b
    out = a + b

    return out

# Max function
def max(a, b):
    # EDIT: Implement a function that return the largest of the two numbers
    # 1. If `a` is greater than `b`, return `a`
    # 2. If `b` is greater than `a`, return `b`
    # Otherwise a and b is equal, and any can be returned
    
    # EDIT: Remove the placeholder
    ...
```"""},
        {"role": "assistant", "content": f"""Sure, here's the code following the EDIT comments

```py
def sum(a: int, b: int) -> int:
    # Return a sum of a +b
    out = a + b

    return out

# Max function
def max(a, b):
    if a > b:
        return a
    if b > a:
        return b

    return a
```"""},
        {"role": "user", "content": f"""
## Problem
>{instruction}

## Code
```py
{annotated_code}
```"""},
    ]

class ChineseWallModel(EditModel):
    def __init__(
        self,
        architect: ChatModel,
        editor: EditModel,
        architect_prompt: ArchitectPromptFunction = architect_prompt_1shot,
    ):
        super().__init__()
        self.architect = architect
        self.editor = editor
        self.architect_prompt = architect_prompt
        self.architect_code_extractor = python_markdown_codeblock_extract

    def generate(self, prompts: List[EditCommand], **kwargs) -> List[EditResponse]:
        assert all(p == prompts[0] for p in prompts), "All prompts must be equal"

        architect_code, architect_resp = self.generate_architect_prompt(prompts[0])

        out = self.editor.generate([{
            **prompts[0],
            "content": architect_code,
        }] * len(prompts), **kwargs)

        return [
            EditResponse(
                content=item["content"],
                instruction=architect_resp,
            ) for item in out
        ]

    def generate_architect_prompt(self, prompt: EditCommand) -> Tuple[str, str]:
        # Architect only gets one prompt
        architect_resp = self.architect.generate(
            [self.architect_prompt(prompt)],
            temperature=0.2,
            top_p=0.95,
            # Gemini 2.5 Pro will not return response if length limit is hit
            max_tokens=100000,
        )

        assert len(architect_resp) > 0  # TODO: Error handling
        architect_code = self.architect_code_extractor(prompt["content"], architect_resp[0])

        return architect_code, architect_resp[0]

class CachedChineseWallModel(ChineseWallModel):
    file_section_marker = b"\n===============\n"

    def __init__(self, *args, cache_path: Optional[Path]=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.cache_hasher = lambda val: blake2b(val, digest_size=8)
        if cache_path:
            self.cache_path = Path(cache_path)
        else:
            self.cache_path = Path(__file__).parent.parent / "chinesewall-cache"

    def generate_architect_prompt(self, prompt: EditCommand) -> Tuple[str, str]:
        # XXX: This doesn't do in memory cache
        if not self.cache_path.exists():
            self.cache_path.mkdir()

        cache_file = self.cache_path / (self.get_cache_key(prompt) + ".txt")
        if cache_file.exists():
            with cache_file.open("rb") as fp:
                return self.parse_file(fp)

        architect_code, architect_resp = super().generate_architect_prompt(prompt)
        with cache_file.open("wb") as fp:
            self.write_file(fp, (architect_code, architect_resp))

        return architect_code, architect_resp

    def get_cache_key(self, prompt: EditCommand):
        # XXX: This doesn't address the change in models, params, etc.
        generated_prompt = self.architect_prompt(prompt)
        return prompt.get("name", "") + "_" + prompt.get("kind", "") + "_" + self.cache_hasher(repr(generated_prompt).encode("utf8")).hexdigest()

    def parse_file(self, fp: io.BufferedIOBase) -> Tuple[str, str]:
        # This uses custom file format
        #
        # The first line is an array of length in bytes of each block, which there must be two (eg. [1,2])
        # After the new line, the first block is read for the number of bytes specified
        # Then it is followed by the file_section_marker
        # Then the second block is read in the same way, and so on.
        # If the block is string, it is utf8 encoded.
        #
        # The remainder of the file after the last block (which there is no trailing file_section_marker) are considered garbage
        #
        # The reason this custom format is used is because it allows human review of the cached data without special character encoding, etc.
        first_line = fp.readline()
        code_len, response_len = json.loads(first_line)
        code = fp.read(code_len).decode("utf8")
        assert fp.read(len(self.file_section_marker)) == self.file_section_marker
        response = fp.read(response_len).decode("utf8")

        return code, response

    def write_file(self, fp: io.BufferedIOBase, data: Tuple[str, str]):
        fp.write(json.dumps([len(data[0]), len(data[1])], ensure_ascii=True).encode("ascii"))
        fp.write(b"\n")
        fp.write(data[0].encode("utf8"))
        fp.write(self.file_section_marker)
        fp.write(data[1].encode("utf8"))
        fp.write(b"\n")
