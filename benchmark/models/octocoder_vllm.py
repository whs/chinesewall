from typing import List

from .chat_base import ChatModel, Message


class OctoCoderChatModel(ChatModel):
    def __init__(
        self,
        model_name="bigcode/octocoder",
        num_gpus=1,
        gpu_util=0.95,
        quantization=False,
    ):
        self.model = LLM(
            model_name,
            dtype=autodetect_dtype() if not quantization else "float16",
            tensor_parallel_size=num_gpus,
            gpu_memory_utilization=gpu_util,
            quantization="awq" if quantization else None,
        )
        self.tokenizer = vllm_get_tokenizer(self.model)

    def fmt_msg(self, message: List[Message]) -> str:
        fmt = []
        start = 0
        system = ""

        # check for system prompt
        if message[0]["role"] == "system":
            start = 1
            system = message[0]["content"]

        for i in range(start, len(message)):
            current = message[i]
            assert current["content"] is not None, "Content of a message cannot be null"
            assert current["role"] is not None, "Role of a message cannot be null"
            if current["role"] == "user":
                # if question, then add system prompt
                fmt.append(f"Question: {system}\n{current['content']}")
                # if last message and is a question, add an answer to it
                if i == len(message) - 1:
                    fmt.append(f"Answer:")
            else:
                # if answer, then no system prompt
                fmt.append(f"Answer: {current['content']}")

        return "\n\n".join(fmt)

    def generate(self, messages: List[List[Message]], **kwargs) -> List[str]:
        kwargs_gen = kwargs.copy()

        msgs = [self.fmt_msg(msg) for msg in messages]

        stop = kwargs_gen.pop("stop", [])
        stop.append("\n\nAnswer:")
        stop.append("\n\nQuestion:")
        # stop.append("\n\n")

        gens = self.model.generate(
            prompts=msgs,
            sampling_params=SamplingParams(
                top_p=kwargs_gen.pop("top_p", 0.95),
                temperature=kwargs_gen.pop("temperature", 0.2),
                max_tokens=kwargs_gen.pop("max_tokens", 1024),
                stop=list(set(stop)),
                **kwargs_gen,
            ),
            use_tqdm=True,
        )

        responses = []

        for gen in gens:
            out = gen.outputs[0].text
            responses.append(out)

        return responses
