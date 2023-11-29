# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0


from transformers import AutoModelForCausalLM, AutoTokenizer 
import os 

os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"

checkpoint = "bigcode/starcoder"

device = "cuda" # for GPU usage or "cpu" for CPU usage 

tokenizer = AutoTokenizer.from_pretrained(checkpoint)

model = AutoModelForCausalLM.from_pretrained(checkpoint).to(device)

inputs = tokenizer("def add(a, b):", return_tensors="pt").input_ids.to(device)

outputs = model.generate(inputs)

print(tokenizer.decode(outputs[0], clean_up_tokenization_spaces=False))