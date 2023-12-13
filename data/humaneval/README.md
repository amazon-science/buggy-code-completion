# Buggy-HumanEval
Buggy-HumenEval contains buggy-code completion task instances constructed form injecting operator flips to reference solutions of [HumanEval problems](https://github.com/openai/human-eval). See more details in [our paper](../../README.md).

## Instructions

Placed the following files under `./problems`.

- `HumanEval.jsonl.gz` can be found from the openai/human-eval repository [here](https://github.com/openai/human-eval/blob/463c980b59e818ace59f6f9803cd92c749ceae61/data/HumanEval.jsonl.gz).

- `humaneval_large_instances.jsonl.gz` is already under `./humaneval/problems` and follows [the MIT license](./humaneval/problems/LICENSE).