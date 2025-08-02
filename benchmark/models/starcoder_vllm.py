from typing import List

from .edit_base import EditModel, EditCommand, EditResponse


def starcoder_edit_prompt(old, instr, new):
    # starcoder tokens
    OLD_CODE_TOKEN = "<commit_before>"
    REFLECTION_TOKEN = "<commit_msg>"
    NEW_CODE_TOKEN = "<commit_after>"
    return OLD_CODE_TOKEN + old + REFLECTION_TOKEN + instr + NEW_CODE_TOKEN + new


class StarCoderCommitEditModel(EditModel):
    def __init__(
        self,
        model_name="bigcode/starcoderbase",
        num_gpus=1,
        before_content_tok="<commit_before>",
        instruction_tok="<commit_msg>",
        after_content_tok="<commit_after>",
    ):
        super().__init__(before_content_tok, instruction_tok, after_content_tok)
        self.model = LLM(
            model_name,
            dtype=autodetect_dtype(),
            tensor_parallel_size=num_gpus,
        )
        self.tokenizer = vllm_get_tokenizer(self.model)
        self.instruction_tok_id = self.tokenizer.encode(instruction_tok)[0]
        self.after_content_tok_id = self.tokenizer.encode(after_content_tok)[0]
        self.after_content_tok = after_content_tok

    def generate(self, prompts: List[EditCommand], **kwargs) -> List[EditResponse]:
        str_prompts = []

        for prompt in prompts:
            content = (
                ("\n" + prompt["content"] +
                 "\n") if prompt["content"] != "" else ""
            )
            if prompt["instruction"] is not None:
                str_prompt = f"{self.before_content_tok}{content}{self.instruction_tok}\n{prompt['instruction']}\n{self.after_content_tok}"
                if "declaration" in kwargs:
                    str_prompt += f"\n{kwargs['declaration']}"
            else:
                str_prompt = f"{self.before_content_tok}{content}{self.instruction_tok}"

            str_prompts.append(str_prompt)

        # generate
        kwargs = kwargs.copy()
        stop = kwargs.pop("stop", [])
        # TODO: double check this
        kwargs["stop"] = stop + [self.after_content_tok]
        gens = self.edit_model_generate(self.model, str_prompts, **kwargs)

        responses = []

        for prompt, gen in zip(prompts, gens):
            out = gen.outputs[0].token_ids

            resp = {"content": "", "instruction": None}
            # if we had an instruction, we are all good.
            # or, it could be that the model didn't generate anything useful
            if (
                prompt["instruction"] is not None
                or self.after_content_tok_id not in out
            ):
                resp["content"] = self.tokenizer.decode(
                    out, skip_special_tokens=True)
                responses.append(resp)
                continue

            # otherwise, find the end of the instruction
            new_content_idx = out.index(self.after_content_tok_id)
            resp["instruction"] = self.tokenizer.decode(
                out[:new_content_idx], skip_special_tokens=True
            )
            # and decode the content
            resp["content"] = self.tokenizer.decode(
                out[new_content_idx + 1:], skip_special_tokens=True
            )
            responses.append(resp)

        return responses

    def get_prompt_format(self):
        return starcoder_edit_prompt
