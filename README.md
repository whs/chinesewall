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

## Paper
The paper code is written in Typst. It vendorized the following code

- [Citation Styles](https://github.com/citation-style-language/styles/blob/master/ieee-with-url.csl) Copyright CC BY-SA 3.0 (see list of authors in the file)
- [arkheion template](https://github.com/mgoulao/arkheion) Copyright MIT License Manuel Goulão

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
