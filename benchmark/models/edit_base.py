from typing import List, Optional, TypedDict, Callable

from CanItEdit.benchmark.models.chat_base import ChatModel, Message
from CanItEdit.benchmark.models.openai import openai_edit_prompt_1shot

# (old, instr) -> [messages]
MessagesFormatFunction = Callable[[str, str], List[Message]]

# (old, new) -> response
PostProcessFunction = Callable[[str, str], str]

class EditCommand(TypedDict):
    name: Optional[str]
    kind: Optional[str]
    instruction: Optional[str]
    content: str


class EditResponse(TypedDict):
    instruction: Optional[str]
    content: str


class EditModel:
    def __init__(self, before_content_tok=None, instruction_tok=None, after_content_tok=None):
        self.before_content_tok = before_content_tok
        self.instruction_tok = instruction_tok
        self.after_content_tok = after_content_tok

    def get_before_content_tok(self) -> Optional[str]:
        return self.before_content_tok

    def get_instruction_tok(self) -> Optional[str]:
        return self.instruction_tok

    def get_after_content_tok(self) -> Optional[str]:
        return self.after_content_tok

    def generate(self, prompts: List[EditCommand], **kwargs) -> List[EditResponse]:
        raise NotImplementedError

    def bugfix_instr(self, prompt) -> Optional[str]:
        return None

    def get_prompt_format(self):
        raise NotImplementedError

    def get_tokenizer(self):
        raise NotImplementedError


def python_markdown_codeblock_extract(_: str, new: str) -> str:
    lines = new.split("\n")
    buf = ""
    in_codeblock = False
    for ln in lines:
        if ln.startswith("```"):
            if in_codeblock:
                break
            else:
                in_codeblock = True
        elif in_codeblock:
            buf += ln + "\n"
    return buf


class ChatAdaptorEditModel(EditModel):
    """
    This is an adaptor class to use ChatModels as EditModels.
    NOTE: This model class is only intended for inference, not training.
    """

    # TODO: implement whole shebang for bugfix

    def __init__(
        self,
        chat_model: ChatModel,
        prompt_format: MessagesFormatFunction = openai_edit_prompt_1shot,
        post_process: PostProcessFunction = python_markdown_codeblock_extract,
    ):
        super().__init__()
        self.model = chat_model
        self.prompt_format = prompt_format
        self.post_process = post_process

    def generate(self, prompts: List[EditCommand], **kwargs) -> List[EditResponse]:
        kwargs = kwargs.copy()
        # TODO: can do something with declaration here
        kwargs.pop("declaration", None)

        kwargs.pop("use_tqdm", None)

        chat_prompts = []
        for prompt in prompts:
            assert (
                prompt["instruction"] is not None
            ), "Every command must have an instruction in ChatAdaptorEditModel"
            chat_prompts.append(
                self.prompt_format(prompt["content"], prompt["instruction"])
            )

        # generate
        gens = self.model.generate(chat_prompts, **kwargs)

        responses = []
        for prompt, gen in zip(prompts, gens):
            processed = self.post_process(prompt["content"], gen)
            resp = {"content": processed, "instruction": None}
            responses.append(resp)

        return responses
