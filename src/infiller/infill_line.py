# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0


import torch
import io, os
import tokenize
from tqdm import tqdm
from utils.provider import load_task
from utils.helpers import separate_code, remove_lines, trim_line
from infiller.infill_ops import infill
import numpy as np

arth_ops =  ['+', '-', '*', '/', '+=', '-=', '*=', '/=', '>', '<', '<=', '>=', '!=', '==', '//', '%', '**', '%=', '//=']


def get_score_next_token(codelm, tokenizer, prompt_test, device):
    p = tokenizer(prompt_test, truncation=True, max_length=700, return_tensors='pt')
    p = {key: value.to(device) for key, value in p.items()}
    input_ids=p['input_ids']
    max_length = 1 + input_ids[0].flatten().size(0)
    with torch.no_grad():
        outs = codelm.generate(input_ids=input_ids, max_length=max_length, pad_token_id=tokenizer.eos_token_id, do_sample=True, temperature=0.6, return_dict_in_generate=True, output_scores=True)
    z = outs['scores'][0].softmax(dim=-1)
    return z


def get_score(tokenizer, s, z):
    my_score = 0
    my_token = s
    for indent in ['', ' ']:
        temp_token = s + indent
        temp_id = tokenizer.encode(temp_token)[1]
        temp_score = z[temp_id].item()
        if temp_score > my_score:
            my_score = temp_score
            my_token =  temp_token
    return my_token, my_score

def predict_next(infiller, tokenizer, op, prompt, code_lines, device):
    s, e = op.start[1], op.end[1]
    l = op.start[0] - 1
    prompt_test = prompt + '\n' + '\n'.join(code_lines[:l]) + '\n' + code_lines[l][:s]
    z = get_score_next_token(infiller, tokenizer, prompt_test, device)[0]
    max_idx = z.argmax().item()
    max_token = tokenizer.decode(max_idx).strip().split(' ')[0]
    max_score = z[max_idx].item()
    new_line = code_lines[l][:s] + max_token + ' ' + code_lines[l][e:]
    return max_score, max_token, z, l, new_line


def find_last_input_line(code):
    input_str = 'input()'
    raw_input_str = 'raw_input()'
    input_str_1 = 'input ( )'
    raw_input_str_1 = 'raw_input ( )'
    code_lines = code.split('\n')
    n = len(code_lines) - 1
    while n > 0:
        if input_str in code_lines[n] or raw_input_str in code_lines[n] or raw_input_str_1 in code_lines[n] or input_str_1 in code_lines[n]:
            return n
        n -= 1
    return n


def find_potential_bugs(codelm, tokenizer, prompt, code, device):
    code_lines = code.split('\n')
    tokens = tokenize.generate_tokens(io.StringIO(code).readline)
    ops = [t for t in tokens if (t.type == 54 and t.string in arth_ops)]
    positions = []
    for op in ops:
        s, e = op.start[1], op.end[1]
        l = op.start[0] - 1
        prompt_test = prompt + '\n' + '\n'.join(code_lines[:l]) + '\n' + code_lines[l][:s]
        z = get_score_next_token(codelm, tokenizer, prompt_test, device)[0]
        max_idx = z.argmax().item()
        max_token = tokenizer.decode(max_idx).strip().split(' ')[0]
        max_score = z[max_idx].item()
        # current token
        my_score = 0
        my_token = op.string
        for indent in ['', ' ']:
            temp_token = op.string + indent
            encoded_tokens = tokenizer.encode(temp_token)
            if len(encoded_tokens) > 1:
                temp_id = encoded_tokens[1]
            else:
                temp_id = encoded_tokens[0]
            temp_score = z[temp_id].item()
            if temp_score > my_score:
                my_score = temp_score
                my_token =  temp_token
        new_line = code_lines[l][:s] + max_token + ' ' + code_lines[l][e:]
        
        positions.append((l, new_line, my_token, my_score, max_token, max_score))
    return positions


def localize_line(codelm, tokenizer, prompt, buggy_code, threshold=0.9, device='cuda'):
    positions = find_potential_bugs(codelm, tokenizer, prompt, buggy_code, device)
    # found_l = positions[0][0] if len(positions) > 0 else 0
    found_l = 0
    for r in positions:
        l, new_l, op, op_score, max_op, max_score = r
        if not max_op.startswith(op) and max_score - op_score > threshold:
            # print(found_l, max_score - op_score)
            found_l = l
            break
    if found_l > 0:
        last_line = find_last_input_line(buggy_code) + 1
        if last_line > found_l:
            found_l = last_line
        code_lines = buggy_code.split('\n')
        partial_code =  '\n'.join([prompt] + code_lines[:found_l] + [''])
    else:
        partial_code = None
    return found_l, partial_code


def localize_line_oracle(infiller, infiller_tokenizer, prompt, buggy_code, clean_code, device):
    buggy_code = remove_lines(buggy_code)
    clean_code = remove_lines(clean_code)
    buggy_lines = buggy_code.split('\n')
    clean_lines = clean_code.split('\n')
    n, m = len(buggy_lines), len(clean_lines)
    idx = -1
    for i in range(min(n, m)):
        if buggy_lines[i] != clean_lines[i]:
            idx = i
            break
    if n > m and idx == -1:
        idx = m 

    if idx == -1:
        return prompt
    else:
        # infill 
        str1 = prompt + '\n' + '\n'.join(buggy_lines[:idx]) + '\n'
        str2 = '\n'.join(buggy_lines[idx+1:])
        outs = infill(infiller, infiller_tokenizer, [str1, str2], max_to_generate=16, temperature=0.6, extra_sentinel=True, max_retries=1, device=device)
        texts = trim_line(outs['infills'][0])
        buggy_lines[idx] = texts[0]
        return prompt + '\n' + '\n'.join(buggy_lines)
