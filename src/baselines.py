# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0


def remove_code(prompt, data_name):
    separator1 = '"""'
    separator2 = "'''"
    if separator1 in prompt:
        sep = separator1
    elif separator2 in prompt:
        sep = separator2
    else:
        print("Cannot find separator!!!")
        return None 

    if data_name.lower() == 'fixeval':
        lst_prompt = prompt.split(sep)
        s = lst_prompt[-1]
        prompt = sep.join(lst_prompt[:-1]) + sep
        lst = s.split('\n')
        n = len(lst)
        last_idx = 0
        for i in range(n-1, 0, -1):
            if 'input()' in lst[i]:
                last_idx = i
                break
        txt = prompt + '\n'.join(lst[:last_idx+1])
        return txt
    else:
        lst_prompt = prompt.split(sep)
        s = lst_prompt[-1]
        return sep.join(lst_prompt[:-1]) + sep



def fix_bugs(fixer, text, threshold=0.9):
    res = [k for k in range(len(text)) if text.startswith('\ndef', k)]
    if len(res) >= 2:
        text = text[:res[1]]
    try:
        outs = fixer(text)
        out_text = outs[0]['text']
        prob = outs[0]['prob']
        if prob > threshold:
            return out_text
        else:
            return text
    except:
        out_text = text 
    return out_text
