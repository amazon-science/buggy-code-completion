# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0


import os, json

def localize_bug_line(buggy_code, clean_code):
    buggy_lines = buggy_code.split('\n')
    clean_lines = clean_code.split('\n')
    n, m = len(buggy_lines), len(clean_lines)
    for i in range(min(n, m)):
        if buggy_lines[i] != clean_lines[i]:
            return i
    if n > m:
        return m 
    return -1

def localize_bug_span(buggy_text, clean_text):
    for i in range(len(buggy_text)):
        if buggy_text[i] != clean_text[i]:
            start_index = i
            j = i + 1
            while j < len(buggy_text) and buggy_text[j] != ' ':
                j += 1
            
            if j < len(buggy_text):
                end_index = j - 1
                return start_index, end_index
            return -1, -1
    return -1, -1

def try_mkdir(folder):
    if not os.path.exists(folder):
        os.makedirs(folder)


def logging(logf, msg, console_print=True):
    logf.write(msg + '\n')
    if console_print:
        print(msg)


def load_results(resul_file):
    samples = []
    with open(resul_file) as fp:
        lines = fp.readlines()
        for l in lines:
            sample = json.loads(l)
            samples.append(sample)
    return samples

def truncate_str(gen_text):
    truncate_before_pattern=["\n\n\n", r"\n\n^#", '"""', '\n\ndef', "^'''",  "<|endoftext|>", "</", "\nfrom", "\nimport", 'if __name__ ==', '\n\n']
    min_idx = 10000
    for s in truncate_before_pattern:
        if s in gen_text:
            idx = gen_text.find(s)
            if idx > -1 and min_idx > idx:
                min_idx = idx
    if min_idx < 10000:
        return gen_text[:min_idx]
    return gen_text


def truncate_str_all(orig_text, new_text):
    truncate_before_pattern=["\n\n\n", r"\n\n^#",  '\n\ndef', "^'''",  "<|endoftext|>", "</", "\nfrom "]
    min_idx = 10000
    gen_text = new_text[len(orig_text):]
    for s in truncate_before_pattern:
        if s in gen_text:
            idx = gen_text.find(s)
            if idx > -1 and min_idx > idx:
                min_idx = idx
    if min_idx < 10000:
        return new_text[:len(orig_text)] + gen_text[:min_idx]
    return  new_text


def separate_code(text):
    separator1 = '"""'
    separator2 = "'''"

    if separator1 in text:
        sep = separator1
    elif separator2 in text:
        sep = separator2
        
    lst_text = text.split(sep)
    s = lst_text[-1]
    prompt = sep.join(lst_text[:-1]) + sep
    return prompt, s
    

def remove_lines(code, indent=''):
    ls = code.rstrip().split('\n')
    lines = []
    for i in range(len(ls)):
        if ls[i].strip() != '':
            lines.append(ls[i][len(indent):])
    return '\n'.join(lines)

def trim_line(t):
    s = t.split('\n')
    start = 0
    for i in range(len(s)):
        if s[i].strip() != '':
            start = i
            break

    return s[start:]