from typing import Optional, List, Callable, Literal, Union

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from vllm import LLM, SamplingParams, RequestOutput, CompletionOutput

from .edit_base import PostProcessFunction, EditModel, EditCommand, EditResponse

# (old, instr, new) -> prompt
PromptFormatFunction = Callable[[str, str, str], str]

CompletionEngine = Literal["vllm", "transformers"]

def autodetect_dtype() -> str:
    if torch.cuda.is_bf16_supported():
        return "bfloat16"
    else:
        return "auto"

def direct_edit_prompt(
    old,
    instr,
    new,
    codeblock_before: Optional[str] = None,
    codeblock_after: Optional[str] = None,
):
    """
    The codeblock_before and codeblock_after arguments are used to specify
    if there should be a codeblock surrounding the code before and after
    the instruction. If None, then no codeblock is used. The string is the
    extension of the codeblock, e.g. "py" or "md".
    """
    if codeblock_before is not None:
        old = f"```{codeblock_before}\n{old}\n```"
    if codeblock_after is not None:
        new = f"```{codeblock_after}\n{new}\n```"
    before = f"""## Code Before:\n{old}\n"""
    instr = f"""## Instruction:\n{instr}\n"""
    after = f"""## Code After:\n{new}"""
    return before + instr + after


def direct_edit_prompt_1shot(
    old,
    instr,
    new,
):
    p = direct_edit_prompt(old, instr, new)
    shot = """## Code Before:
def add(a, b):
    return a + b
## Instruction:
Add a "sub" function that subtracts two numbers. Also write docstrings for both functions and change a,b to x,y.
## Code After:
def add(x, y):
    \"\"\"Adds two numbers.\"\"\"
    return x + y

def sub(x, y):
    \"\"\"Subtracts two numbers.\"\"\"
    return x - y"""
    p = shot + "\n" + p
    return p


class DirectEditModel(EditModel):
    """
    The direct kind of edit model, this class is supposed to be used either with EditCoder or
    with non-chat models, like foundation models.
    """

    def __init__(
        self,
        model_name="codellama/CodeLlama-34b-hf",
        num_gpus=1,
        gpu_util=0.95,
        prompt_format: PromptFormatFunction = direct_edit_prompt,
        post_process: PostProcessFunction = lambda old, new: new,
        completion_engine: CompletionEngine = "vllm",
        stop_tokens: List[str] = ["\n## ", "## Code After:",
                                  "## Instruction:", "## Code Before:"],
        max_model_len=None,
    ):
        super().__init__()
        self.model = init_completion_engine(
            completion_engine,
            model_name=model_name,
            num_gpus=num_gpus,
            gpu_util=gpu_util,
            max_model_len=max_model_len,
        )
        self.prompt_format = prompt_format
        self.post_process = post_process
        self.stop_tokens = stop_tokens

    def edit_model_generate(
        self,
        model: Union[LLM, "TransformersVLLMAdapter"],
        str_prompts: List[str], **kwargs
    ) -> List[RequestOutput]:
        kwargs_gen = kwargs.copy()
        if "declaration" in kwargs_gen:
            del kwargs_gen["declaration"]
        use_tqdm = kwargs_gen.pop("use_tqdm", False)
        gens = model.generate(
            prompts=str_prompts,
            sampling_params=SamplingParams(
                top_p=kwargs_gen.pop("top_p", 0.95),
                temperature=kwargs_gen.pop("temperature", 0.2),
                max_tokens=kwargs_gen.pop("max_tokens", 1024),
                **kwargs_gen,
            ),
            use_tqdm=use_tqdm,
        )
        return gens

    def generate(self, prompts: List[EditCommand], **kwargs) -> List[EditResponse]:
        str_prompts = []

        for prompt in prompts:
            declaration = kwargs["declaration"] if "declaration" in kwargs else ""
            assert prompt["instruction"] is not None, "Not implemented yet"
            str_prompts.append(
                self.prompt_format(
                    prompt["content"], prompt["instruction"], declaration
                )
            )

        kwargs = kwargs.copy()
        stop = kwargs.pop("stop", [])
        kwargs["stop"] = stop + self.stop_tokens

        # generate
        gens = self.edit_model_generate(self.model, str_prompts, **kwargs)

        responses = []
        for prompt, gen in zip(prompts, gens):
            out = gen.outputs[0].text
            try:
                processed = self.post_process(prompt["content"], out)
            except Exception as e:
                # print full stack trace
                import traceback
                traceback.print_exc()
                print("Error in post processing:", e)
                processed = out
            resp = {"content": processed, "instruction": None}
            responses.append(resp)

        return responses

    def get_prompt_format(self):
        return self.prompt_format

    def get_tokenizer(self):
        return self.model.get_tokenizer()


class TransformersVLLMAdapter:
    def __init__(self, model_name):
        dtype = "auto"
        if torch.cuda.is_bf16_supported():
            dtype = torch.bfloat16
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name, torch_dtype=dtype
        ).cuda()
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            #  padding_side="right",
        )
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

    def generate(
            self,
            prompts: List[str],
            sampling_params: SamplingParams,
            use_tqdm: bool = False,
    ) -> List[RequestOutput]:
        # TODO: support heterogeneous prompts
        assert all(
            p == prompts[0] for p in prompts), "All prompts must be the same -- batched heterogeneous prompts not supported"
        new_tokens = sampling_params.max_tokens
        stop = sampling_params.stop
        with torch.no_grad():
            tokens = self.tokenizer(
                prompts,
                return_tensors="pt",
                #  padding=True,
                #  truncaton=True,
                max_length=self.model.config.max_position_embeddings - new_tokens - 2,
            ).to(self.model.device)
            outputs = self.model.generate(
                **tokens,
                max_new_tokens=new_tokens,
                do_sample=True,
                top_p=sampling_params.top_p,
                temperature=sampling_params.temperature,
                use_cache=True,
                pad_token_id=self.tokenizer.eos_token_id
            )
            #  decoded: List[str] = self.tokenizer.batch_decode(
            #  outputs,
            #  skip_special_tokens=True
            #  )
            decoded = [""] * len(prompts)
            for i, (out, prompt) in enumerate(zip(outputs, tokens["input_ids"])):
                out = out[len(prompt):]
                d: str = self.tokenizer.decode(
                    out, skip_special_tokens=True
                )
                assert isinstance(d, str)
                if stop is not None:
                    for s in stop:
                        found = d.find(s)
                        if found != -1:
                            d = d[:found]
                decoded[i] = d

        decoded_vllm = [RequestOutput(
            request_id="",
            prompt=prompt,
            prompt_token_ids=[],
            prompt_logprobs=None,
            outputs=[CompletionOutput(
                index=0,
                text=dec,
                token_ids=[],
                cumulative_logprob=0.0,
                logprobs=None,
                finish_reason=None,
            )],
            finished=True
        ) for (prompt, dec) in zip(prompts, decoded)]

        return decoded_vllm

    def get_tokenizer(self):
        return self.tokenizer


def init_completion_engine(engine: CompletionEngine, **kwargs):
    if engine == "vllm":
        extra_kwargs = {}
        if "max_model_len" in kwargs:
            extra_kwargs["max_model_len"] = kwargs["max_model_len"]
        return LLM(
            kwargs["model_name"],
            dtype=autodetect_dtype(),
            tensor_parallel_size=kwargs["num_gpus"],
            gpu_memory_utilization=kwargs["gpu_util"],
            **extra_kwargs,
        )
    elif engine == "transformers":
        return TransformersVLLMAdapter(kwargs["model_name"])
    else:
        raise ValueError(f"Unknown completion engine {engine}")

