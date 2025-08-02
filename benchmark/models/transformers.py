import transformers
from typing import List

from tqdm import tqdm

from .edit_base import EditModel, EditCommand, EditResponse
from .direct import PromptFormatFunction, PostProcessFunction, direct_edit_prompt_1shot


# Deprecated: Use TransformersVLLMAdapter
class TransformersEditModel(EditModel):
    def __init__(
        self,
        model_name="starcoder2:instruct",
        prompt: PromptFormatFunction = direct_edit_prompt_1shot,
        post_process: PostProcessFunction = lambda old, new: new,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.tokenizer = transformers.AutoTokenizer.from_pretrained(model_name)
        self.model = transformers.AutoModelForCausalLM.from_pretrained(model_name, device_map="auto")
        self.prompt_format = prompt
        self.post_process = post_process

    def generate(self, prompts: List[EditCommand], **kwargs) -> List[EditResponse]:
        kwargs.setdefault("temperature", 0.75)
        kwargs.setdefault("top_p", 0.9)
        kwargs.setdefault("max_new_tokens", kwargs.pop("max_tokens", 256))
        kwargs.setdefault("stop_strings", ["## Code After:",
                                           "## Instruction:", "## Code Before:"])

        out = []
        for prompt in tqdm(prompts):
            assert (prompt["instruction"] is not None), "Every command must have an instruction in TransformersEditModel"

            declaration = kwargs["declaration"] if "declaration" in kwargs else ""
            model_input = self.prompt_format(prompt["content"], prompt["instruction"], declaration)
            model_input_tokens = self.tokenizer(model_input, return_tensors="pt", return_token_type_ids=False).to("cuda:0") # FIXME

            model_output = self.model.generate(tokenizer=self.tokenizer, **model_input_tokens, **kwargs)
            response = self.tokenizer.decode(model_output[0], skip_special_tokens=True)

            out.append({"content": self.post_process(prompt["content"], response)[len(model_input):]})

        return out

