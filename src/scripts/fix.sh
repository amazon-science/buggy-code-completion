# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

data=humaneval
model=codegen-350M-mono
mode=clean # for this setting only
comp_file=../results/completion/"$data"/"$model"/"$mode"/_naive_completion_100_948-1895.jsonl
if test -f "$comp_file"; then
    echo $model
    python comp_fix.py --sample_file "$comp_file" #--out_file "$out_file"
else
    echo $model 'not found'
fi