# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0


import torch
from typing import List


# signals the start of a document
BOS = "<|endoftext|>"
# signals the end of a generated infill
EOM = "<|endofmask|>"

# print intermediate outputs of infilling
VERBOSE = False

# Incoder code
def make_sentinel(i):
    # signals (1) a location to insert an infill and (2) the start of the infill generation
    return f"<|mask:{i}|>"

# Modified from Incoder code
def generate_infill(model, tokenizer, input: str, max_to_generate: int=128, temperature: float=0.2, device='cuda'):
    input_ids = tokenizer(input, return_tensors="pt").input_ids
    input_ids = input_ids.to(device)
    max_length = max_to_generate + input_ids.flatten().size(0)
    if max_length > 2048:
        print("warning: max_length {} is greater than the context window {}".format(max_length, 2048))
    with torch.no_grad():
        output = model.generate(input_ids=input_ids, do_sample=True, top_p=0.95, temperature=temperature, max_length=max_length)
    # pass clean_up_tokenization_spaces=False to avoid removing spaces before punctuation, e.g. "from ." -> "from."
    detok_hypo_str = tokenizer.decode(output.flatten(), clean_up_tokenization_spaces=False)
    if detok_hypo_str.startswith(BOS):
        detok_hypo_str = detok_hypo_str[len(BOS):]
    return detok_hypo_str

# Modified from Incoder code
def infill(model, tokenizer, parts: List[str], max_to_generate: int=128, temperature: float=0.2, extra_sentinel: bool=True, max_retries: int=1, device='cuda'):
    assert isinstance(parts, list)
    retries_attempted = 0
    done = False

    while (not done) and (retries_attempted < max_retries):
        retries_attempted += 1

        if VERBOSE:
            print(f"retry {retries_attempted}")
        
        ## (1) build the prompt
        if len(parts) == 1:
            prompt = parts[0]
        else:
            prompt = ""
            # encode parts separated by sentinel
            for sentinel_ix, part in enumerate(parts):
                prompt += part
                if extra_sentinel or (sentinel_ix < len(parts) - 1):
                    prompt += make_sentinel(sentinel_ix)
        
        infills = []
        complete = []

        done = True

        ## (2) generate infills
        for sentinel_ix, part in enumerate(parts[:-1]):
            # print(f"\n\nprompt {sentinel_ix}\n", prompt)
            complete.append(part)
            prompt += make_sentinel(sentinel_ix)
            # TODO: this is inefficient as it requires re-encoding prefixes repeatedly
            completion = generate_infill(model, tokenizer, prompt, max_to_generate, temperature, device)
            completion = completion[len(prompt):]
            if EOM not in completion:
                if VERBOSE:
                    print(f"warning: {EOM} not found")
                completion += EOM
                done = False
            completion = completion[:completion.index(EOM) + len(EOM)]
            infilled = completion[:-len(EOM)]
            infills.append(infilled)
            complete.append(infilled)
            prompt += completion
            # print("complete: ", complete)
        complete.append(parts[-1])
        text = ''.join(complete)

    if VERBOSE:
        print("generated text:")
        print(prompt)
        print()
        print("parts:")
        print(parts)
        print()
        print("infills:")
        print(infills)
        print()
        print("restitched text:")
        print(text)
        print()
    
    return {
        'text': text, # str, the completed document (with infills inserted)
        'parts': parts, # List[str], length N. Same as passed to the method
        'infills': infills, # List[str], length N-1. The list of infills generated
        'retries_attempted': retries_attempted, # number of retries used (if max_retries > 1)
    } 
