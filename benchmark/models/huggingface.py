from typing import List

from .chat_base import ChatModel, Message


class HFChatModel(ChatModel):
    B_INST, E_INST = "[INST]", "[/INST]"
    B_SYS, E_SYS = "<<SYS>>\n", "\n<</SYS>>\n\n"

    def __init__(
        self,
        model_name="codellama/CodeLlama-34b-Instruct-hf",
        num_gpus=1,
        gpu_util=0.95,
        quantization=False,
        system_supported=True,
        max_model_len=None,
    ):
        self.model = LLM(
            model_name,
            dtype=autodetect_dtype() if not quantization else "float16",
            tensor_parallel_size=num_gpus,
            gpu_memory_utilization=gpu_util,
            max_model_len=max_model_len,
            quantization="awq" if quantization else None,
        )
        self.tokenizer = vllm_get_tokenizer(self.model)
        self.system_supported = system_supported

    def llama2_chat_generate(
        self, messages: List[List[Message]], **kwargs
    ) -> List[str]:
        def tokenize_messages(ms) -> List[int]:
            ms = ms.copy()
            if not self.system_supported and ms[0]["role"] == "system":
                sys_m = ms[0]["content"]
                ms = ms[1:]
                for m in ms:
                    if m["role"] == "user":
                        m["content"] = sys_m + "\n" + m["content"]

            toks = self.tokenizer.apply_chat_template(
                ms,  # type: ignore
                tokenize=True,
                truncation=True,
                add_generation_prompt=True,
                max_length=16384 - max_new_tokens - 2,
            )
            assert isinstance(toks, list)
            if "prefix_after" in ms[-1]:
                toks.extend(
                    self.tokenizer.encode(
                        ms[-1]["prefix_after"], add_special_tokens=False
                    )
                )

            return toks

        kwargs = kwargs.copy()
        max_new_tokens = kwargs.pop("max_tokens", 256)
        prompts = [tokenize_messages(ms) for ms in messages]

        discard_lengthy = kwargs.pop("discard_lengthy", False)
        use_tqdm = kwargs.pop("use_tqdm", False)
        stop = kwargs.pop("stop", [])
        stop.append(self.E_INST)
        params = SamplingParams(
            top_p=kwargs.pop("top_p", 0.9),
            temperature=kwargs.pop("temperature", 0.75),
            max_tokens=max_new_tokens,
            stop=list(set(stop)),
            **kwargs,
        )
        gens = self.model.generate(
            prompt_token_ids=prompts,
            sampling_params=params,
            use_tqdm=use_tqdm,
        )
        decoded = []
        for ms, gen in zip(messages, gens):
            outs = gen.outputs[0]
            if discard_lengthy and outs.finish_reason == "length":
                continue
            toks = outs.token_ids
            dec = self.tokenizer.decode(toks, skip_special_tokens=True)

            if "prefix_after" in ms[-1]:
                dec = ms[-1]["prefix_after"] + dec

            for s in stop:
                found = dec.find(s)
                if found != -1:
                    dec = dec[:found]

            stripped = dec.strip()
            decoded.append(stripped)
        return decoded

    def generate(self, messages: List[List[Message]], **kwargs) -> List[str]:
        # make sure we have a list of lists
        assert isinstance(messages, list), "messages must be a list of lists"
        assert len(messages) > 0, "messages must have at least one list"
        assert isinstance(
            messages[0], list), "messages must be a list of lists"
        return self.llama2_chat_generate(messages, **kwargs)
