# Third-party material

contio itself is licensed under MIT OR Apache-2.0 (see `LICENSE-MIT`, `LICENSE-APACHE`).
The benchmark data in `registry/bench/` includes small subsets of public datasets,
which keep their own licenses. The subsets are produced by `scripts/bench/fetch_subsets.py`.
`retrieval.json` and `tool_calls.json` are written for contio and fall under contio's license.

| File | Source | License |
|---|---|---|
| `reasoning_gsm8k.json` | [GSM8K](https://github.com/openai/grade-school-math), OpenAI (test split, sampled) | MIT |
| `coding_humaneval.json` | [HumanEval](https://github.com/openai/human-eval), OpenAI (sampled) | MIT |
| `knowledge_mmlu_pro.json` | [MMLU-Pro](https://huggingface.co/datasets/TIGER-Lab/MMLU-Pro), TIGER-Lab (test split, sampled) | MIT |
| `speech.json`, `speech/*.flac` | [LibriSpeech ASR corpus](https://www.openslr.org/12) by Vassil Panayotov, Guoguo Chen, Daniel Povey and Sanjeev Khudanpur, via [openslr/librispeech_asr](https://huggingface.co/datasets/openslr/librispeech_asr) | [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/) |

The MIT-licensed subsets are used under the MIT license terms (the text is in `LICENSE-MIT`)
with these copyright notices:

- GSM8K, HumanEval: Copyright (c) OpenAI
- MMLU-Pro: Copyright (c) TIGER-Lab

LibriSpeech: 20 utterances from `test-clean`, unchanged audio and transcripts; the selection is ours.
Based on: V. Panayotov, G. Chen, D. Povey, S. Khudanpur, "LibriSpeech: an ASR corpus based on
public domain audio books", ICASSP 2015.

Inference engines (llama.cpp, whisper.cpp, FastFlowLM) are not part of contio: they are
downloaded or found on the system and run as separate programs under their own licenses.
