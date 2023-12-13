# Large Language Models of Code Fail at Completing Code with Potential Bugs


## Overview

This folder contains implementations and scripts for the NeurIPS 2023 paper "[Large Language Models of Code Fail at Completing Code with Potential Bugs](https://arxiv.org/abs/2306.03438)".

## Usage

### Installation 

```
pip install transformers==4.14.1
pip install accelerate
pip install fire
pip install code-tokenize
```

### Datasets
See instructions for [buggy-HumanEval](./data/humaneval/) and [buggy-FixEval](./data/fixeval) datasets under the `data` folder.

### Source codes 
All source codes are stored in the ```src/``` folder
For the baseline, download the Realit package (NBFModel) and put under the ```src/```

### Run experiments

To run completion, use 
```python completor.py --dataset humaneval --mode buggy --model_name codegen-2B-mono --method completion --batch_size 15 --large 1``` 

or use the wandb: 
- modify the configurations in ```sweep_config.yml``` and copy sweep id into run.sh
- ```./run.sh```

Here, 
- ```dataset```: the name of dataset, either ```humameval``` or ```fixeval```
- ```large```: version of dataset, 1 for large and 0 for small
- ```mode```: setting of the partial code, ```clean``` or ```buggy```
- ```model_name```: name of code language models, [`codegen-2B-mono`, `codegen-350M-mono`, `incoder-6B`, `incoder-1B`]
- ```method```: [`completion`, `removal`, `comp_fix`, `infill_line`] for four methods in our paper



#### Evaluation
To evaluate, use the `scripts/eval.sh` script.
For example, here is the script to  evaluate the `pass@k` of `buggy` completion on large `buggy-HumanEval` with `codegen-2B-mono` using `removal-completion` method:
    ```
    sh scripts/eval.sh 0 0-1903 buggy codegen-2B-mono removal
    ``` 


### Get results
Results will be saved under the directory ```results/evals/```<br>
