import gzip
import json
from pathlib import Path
from typing import List, Callable
from typing import TypeVar

import datasets
from tqdm import tqdm

from models.edit_base import EditModel, EditCommand


def gunzip_json_write(path: Path, data: dict) -> None:
    with gzip.open(path, "wt") as f:
        json.dump(data, f)


T = TypeVar("T")


def batch_prompts_from_example(example: T, batch_size: int, completion_limit: int) -> List[List[T]]:
    prompts = [example] * completion_limit
    num_batches = completion_limit // batch_size
    batches = [prompts[i * batch_size: (i + 1) * batch_size]
               for i in range(num_batches)]
    # the last batch may be smaller
    if len(prompts) % batch_size != 0:
        batches.append(prompts[num_batches * batch_size:])

    return batches


# NOTE: this is the factory for each model type. to add a new model type, add a new case here
# and implement it in models.py. Also, add a new case in the argument parser below.
def model_factory(
        model_type: str,
        quantize=False,
        num_gpus=1,
        system_supported=True,
        max_model_len=None,
        **kwargs,
) -> Callable[[str], EditModel]:
    if model_type == "direct":
        from models.direct import DirectEditModel

        return (lambda path: DirectEditModel(
            path,
            completion_engine="vllm",
            num_gpus=num_gpus,
            max_model_len=max_model_len,
        ))
    elif model_type == "direct-1shot":
        from models.direct import DirectEditModel, direct_edit_prompt_1shot

        return (lambda path: DirectEditModel(
            path,
            completion_engine="vllm",
            num_gpus=num_gpus,
            max_model_len=max_model_len,
            prompt_format=direct_edit_prompt_1shot,
        ))
    elif model_type == "starcoder2":
        from models.edit_base import python_markdown_codeblock_extract
        from models.direct import DirectEditModel
        from models.starcoder2 import starcoder2_edit_prompt_1shot

        return (lambda path: DirectEditModel(
            path,
            completion_engine="vllm",
            num_gpus=num_gpus,
            max_model_len=max_model_len,
            prompt_format=starcoder2_edit_prompt_1shot,
            # TODO: fix the hack below
            post_process=(
                lambda x, y: python_markdown_codeblock_extract(x, "```py\n" + y)),
        ))
    elif model_type == "starcoder":
        from models.starcoder_vllm import StarCoderCommitEditModel

        return StarCoderCommitEditModel
    elif model_type == "openai":
        from models.edit_base import ChatAdaptorEditModel
        from CanItEdit.benchmark.models.openai import OpenAIChatModel

        return (lambda path: ChatAdaptorEditModel(OpenAIChatModel(path)))
    elif model_type == "ollama":
        from models.edit_base import ChatAdaptorEditModel
        from CanItEdit.benchmark.models.ollama import OllamaMarkdownChatModel
        from models.starcoder2 import starcoder2_edit_prompt_1shot_chat

        return lambda path: ChatAdaptorEditModel(OllamaMarkdownChatModel(path), prompt_format=starcoder2_edit_prompt_1shot_chat)
    elif model_type == "chinesewall":
        from models.chinesewall import CachedChineseWallModel, editor_prompt_with_instr_1shot
        from models.edit_base import ChatAdaptorEditModel
        from models.ollama import OllamaMarkdownChatModel
        from models.openai import OpenAIChatModel

        return lambda path: CachedChineseWallModel(
            OpenAIChatModel("google/gemini-2.5-pro"),
            ChatAdaptorEditModel(OllamaMarkdownChatModel(path), prompt_format=editor_prompt_with_instr_1shot),
            cache_path=kwargs.get('cache_path', None),
        )
    elif model_type == "chinesewall-direct":
        from models.chinesewall import CachedChineseWallModel
        from models.direct import DirectEditModel
        from models.openai import OpenAIChatModel

        def chinesewall_direct(path: str):
            out = CachedChineseWallModel(
                OpenAIChatModel("google/gemini-2.5-pro"),
                DirectEditModel(
                    path,
                    completion_engine="vllm",
                    num_gpus=num_gpus,
                    max_model_len=max_model_len,
                ),
                cache_path=kwargs.get('cache_path', None),
            )
            out.editor_code_extractor = lambda old, new: new

            return out

        return chinesewall_direct
    elif model_type == "chat":
        from models.edit_base import ChatAdaptorEditModel
        from models.huggingface import HFChatModel

        return (lambda path: ChatAdaptorEditModel(HFChatModel(
            path,
            quantization=quantize,
            num_gpus=num_gpus,
            system_supported=system_supported,
        )))
    elif model_type == "transformers":
        from models.transformers import TransformersEditModel

        return (lambda path: TransformersEditModel(
            path,
        ))
    elif model_type == "octocoder":
        from models.edit_base import ChatAdaptorEditModel
        from models.octocoder_vllm import OctoCoderChatModel

        return (lambda path: ChatAdaptorEditModel(OctoCoderChatModel(
            path,
            quantization=quantize,
            num_gpus=num_gpus,
        )))
    else:
        raise ValueError(f"Unknown model type: {model_type}")


def complete_problem(problem_name: str, example: EditCommand, model: EditModel, batch_size: int, completion_limit: int, **kwargs) -> List[str]:
    batches = batch_prompts_from_example(example, batch_size, completion_limit)

    completions = []
    for batch in tqdm(batches, desc=problem_name):
        resps = model.generate(batch, **kwargs)
        for resp in resps:
            completions.append(resp["content"])

    return completions


def main(args):
    dataset = datasets.load_dataset(
        args.dataset, args.subset, split=args.split)
    model = model_factory(
        args.model_type,
        quantize=args.quantize,
        num_gpus=args.num_gpus,
        system_supported=not args.no_system,
        max_model_len=args.max_model_len,

        cache_path=args.cache_path,
    )(args.model)
    model_kwargs = {
        "temperature": args.temperature,
        "top_p": args.top_p,
        "max_tokens": args.max_tokens,
    }

    if not Path(args.output_dir).exists():
        Path(args.output_dir).mkdir(parents=True)

    # writing in this format such that we can use the MultiPL-E evaluation container :)
    for ex in tqdm(dataset, total=len(dataset)):  # type: ignore
        assert isinstance(ex, dict)

        if args.humanevalpack:
            instr_kinds = ['instruction']
        else:
            instr_kinds = ['instruction_descriptive', 'instruction_lazy']

        for instr_kind in instr_kinds:
            path = Path(args.output_dir) / \
                (f"{ex['full_name']}_{instr_kind}.json.gz")
            if path.exists():
                continue  # this pretty much resumes from where it left off

            instr = ex[instr_kind]
            example = EditCommand(
                name=ex['full_name'],
                kind=instr_kind,
                instruction=instr,
                content=ex["before"],
            )

            if "declaration" in ex:
                model_kwargs["declaration"] = ex["declaration"]

            completions = complete_problem(
                ex['full_name'],
                example,
                model,
                args.batch_size,
                args.completion_limit,
                **model_kwargs,
            )

            # copy over the example
            result = {}
            for k in ex:
                result[k] = ex[k]

            result["instr_kind"] = instr_kind
            # this is for compatibility with the MultiPL-E evaluator
            result["prompt"] = ex["declaration"] if "declaration" in ex else ""
            result["completions"] = completions
            result["language"] = "py"
            result["temperature"] = args.temperature
            result["top_p"] = args.top_p
            result["max_tokens"] = args.max_tokens
            result["stop_tokens"] = []
            result["script_args"] = args.__dict__.copy()

            gunzip_json_write(path, result)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str,
                        default="nuprl/CanItEdit", help="dataset to use")
    parser.add_argument("--split", type=str, default="test",
                        help="split of the dataset to use")
    parser.add_argument("--subset", type=str, default=None,
                        help="subset of the split to use")
    parser.add_argument(
        "--model-type",
        type=str,
        default="direct",
        choices=["direct", "direct-1shot", "chinesewall", "chinesewall-direct",
                 "openai", "chat", "octocoder", "starcoder", "starcoder2", "ollama",
                 "transformers"],
        help="type of model to use for completions",
    )
    parser.add_argument("--model", type=str, required=True,
                        help="path to model or hub name")
    parser.add_argument("--cache-path", type=str,
                        help="path to cache (for chinesewall)")
    parser.add_argument("--output-dir", type=str, required=True,
                        help="output directory for completions")
    parser.add_argument("--batch-size", type=int, default=20,
                        help="batch size for completions")
    parser.add_argument("--completion-limit", type=int,
                        default=20, help="number of completions per prompt")
    parser.add_argument("--temperature", type=float,
                        default=0.2, help="sampling temperature")
    parser.add_argument("--quantize", action="store_true",
                        help="quantize the model with AWQ")
    parser.add_argument("--humanevalpack", action="store_true",
                        help="run humanevalpack instead of CanItEdit")
    parser.add_argument("--top-p", type=float,
                        default=0.95, help="top-p sampling")
    parser.add_argument("--max-tokens", type=int,
                        default=2048, help="max new tokens to generate per completion. 2048 works for CanItEdit")
    parser.add_argument("--max-model-len", type=int, default=None,
                        help="max model length for batching with vLLM. only change if getting OOM")
    parser.add_argument("--num-gpus", type=int, default=1,
                        help="number of gpus for sharded model")
    parser.add_argument("--no-system", action="store_true",
                        help="disable system prompt for chat models")
    args = parser.parse_args()
    main(args)
