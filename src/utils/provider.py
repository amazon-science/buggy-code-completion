# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
# Modifications Copyright 2021 from HumanEval code https://github.com/openai/human-eval


import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from utils.helpers import separate_code, remove_lines
from typing import Iterable, Dict
import gzip
import json
import os

def read_tasks(evalset_file) -> Dict[str, Dict]:
    return {task["task_id"]: task for task in stream_jsonl(evalset_file)}

def read_problems(evalset_file) -> Dict[str, Dict]:
    return [instance for instance in stream_jsonl(evalset_file)]

# HumanEval code
def stream_jsonl(filename: str) -> Iterable[Dict]:
    """
    Parses each jsonl line and yields it as a dictionary
    """
    if filename.endswith(".gz"):
        with open(filename, "rb") as gzfp:
            with gzip.open(gzfp, 'rt') as fp:
                for line in fp:
                    if any(not x.isspace() for x in line):
                        yield json.loads(line)
    else:
        with open(filename, "r") as fp:
            for line in fp:
                if any(not x.isspace() for x in line):
                    yield json.loads(line)

# HumanEval code
def write_jsonl(filename: str, data: Iterable[Dict], append: bool = False):
    """
    Writes an iterable of dictionaries to jsonl
    """
    if append:
        mode = 'ab'
    else:
        mode = 'wb'
    filename = os.path.expanduser(filename)
    if filename.endswith(".gz"):
        with open(filename, mode) as fp:
            with gzip.GzipFile(fileobj=fp, mode='wb') as gzfp:
                for x in data:
                    gzfp.write((json.dumps(x) + "\n").encode('utf-8'))
    else:
        with open(filename, mode) as fp:
            for x in data:
                fp.write((json.dumps(x) + "\n").encode('utf-8'))
    

def load_data(data_dir, data_name, large=True):

    suffix = 'large' if large else 'small'
    if data_name == 'humaneval':
        max_problem = 1895 if large else 524
    else:     
        max_problem = 291 if large else 89
    evalset_file = f'{data_dir}/{data_name}/problems/{data_name}_{suffix}_instances.jsonl.gz'
    return max_problem, read_problems(evalset_file)



def load_task(problems, task_id):
    text = problems[task_id]['prompt']
    prompt, code = separate_code(text)
    code = remove_lines(code, '')
    prompt = remove_lines(prompt, '')
    return prompt, code


def load_model(model_name, device):
    # model
    if model_name.startswith("codegen"):
        checkpoint = f'Salesforce/{model_name}'
    else:
        checkpoint = f"facebook/{model_name}"

    if '6B' in model_name:
        model = AutoModelForCausalLM.from_pretrained(checkpoint, revision="float16", torch_dtype=torch.float16, low_cpu_mem_usage=True)
        model = model.half().to(device)
    else:
        model = AutoModelForCausalLM.from_pretrained(checkpoint)
        model = model.to(device)

    tokenizer = AutoTokenizer.from_pretrained(checkpoint)
    model.eval()
    return model, tokenizer
    
    




    
