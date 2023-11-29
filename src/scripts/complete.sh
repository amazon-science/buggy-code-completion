# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

data=humaneval
model=codegen-350M-mono
CUDA_VISIBLE_DEVICES=0 python src/completor.py --dataset humaneval --mode clean --method rewriting --model_name $model  --part_id 0  --part_sz 30 --use_wandb 0 --large 1