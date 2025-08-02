# Applying the Chinese Wall Reverse Engineering Technique to Large Language Model Code Editing

This repository contains code from my paper ["Applying the Chinese Wall Reverse Engineering Technique to Large Language Model Code Editing". (arXiv:2507.15599)](https://arxiv.org/abs/2507.15599).

## Benchmark
The benchmark is based on [CanItEdit](https://github.com/nuprl/CanItEdit/) by Federico Cassano, et al.

The code is refactored to improve modularity, but new code use hardcoded inputs. To run

```sh
uv sync

# Generate results

# Chinese wall vLLM
python benchmark/generate_completions.py --model=/path/to/comma-v0.1-1t-bnb-8b --model-type=chinesewall-direct --output-dir=outputs-chinesewall-comma --completion-limit=20 --batch-size=5 --temperature=0.2 --max-tokens=2048 --top-p=0.95
# Chinese wall Ollama
python benchmark/generate_completions.py --model=phi4 --model-type=chinesewall --output-dir=outputs-chinesewall-phi4 --completion-limit=20 --batch-size=1 --temperature=0.2 --max-tokens=2048 --top-p=0.95
# vLLM
python benchmark/generate_completions.py --model=/path/to/comma-v0.1-1t-bnb-8b --model-type=direct --output-dir=outputs-comma --completion-limit=20 --batch-size=5 --temperature=0.2 --max-tokens=2048 --top-p=0.95
# OpenAI
OPENAI_API_KEY=... OPENAI_BASE_URL=https://router.requesty.ai/v1 python benchmark/generate_completions.py --model=google/gemini-2.5-pro --model-type=openai --output-dir=outputs-gemini-2.5-pro --completion-limit=20 --batch-size=8 --temperature=0.2 --max-tokens=100000 --top-p=0.95

# Generate result

./benchmark/evaluate_completions.sh /full/path/to/outputs-comma
./benchmark/separate_results.sh outputs-comma
python ./benchmark/pass_k.py outputs-comma_* -k 20
```

### License
Unknown - The original benchmark did not have copyright metadata.

For my own contributions, they are licensed under the MIT license.

## Paper
The paper code is written in Typst. It vendorized the following code

- [Citation Styles](https://github.com/citation-style-language/styles/blob/master/ieee-with-url.csl) Copyright CC BY-SA 3.0 (see list of authors in the file)
- [arkheion template](https://github.com/mgoulao/arkheion) Copyright MIT License Manuel Goulão

The contents of the paper and its accompanying source file are licensed under the [CC-BY 4.0 International](https://creativecommons.org/licenses/by/4.0/) license.

## Output
All outputs produced during the production of the paper are located in the outputs directory. The chinesewall-cache folder is the cache from Gemini 2.5 Pro, and the file format is described in benchmark/models/chinesewall.py in parse_file function.

As discussed in the paper, it is to my believe that the outputs of the LLM may include copyrighted materials and the copyright owners of the files are unclear. Parts of the file are coming from the CanItEdit benchmark and are copyrighted by the original authors. **If possible**, I license the files under [CC0](https://creativecommons.org/public-domain/cc0/) and make no claim of the copyright of the contents.

This includes the following contents:

- All LLM generated contents with unclear license (if it is allowed for me to make a copyright claim on it), except for fields that are copied from the benchmark
- Results from evaluating the accompanying test on the LLM code
- The contents of the chinesewall-cache folder are CanItEdit input source, modified with Gemini 2.5 Pro. The [Vertex AI service specific terms](https://cloud.google.com/terms/service-terms) as of 2025-08-02 states that Google does not claims copyright to the new output, and provide indemnification obligation.

## Citation
```
@misc{hanmongkolchai2025applyingchinesewallreverse,
      title={Applying the Chinese Wall Reverse Engineering Technique to Large Language Model Code Editing}, 
      author={Manatsawin Hanmongkolchai},
      year={2025},
      eprint={2507.15599},
      archivePrefix={arXiv},
      primaryClass={cs.SE},
      url={https://arxiv.org/abs/2507.15599}, 
}
```
