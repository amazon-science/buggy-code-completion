# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

#!/bin/bash

data=$1
range=$2
mode=$3
model_name=$4
fixer=$5
group=$6
by_instance=0
by_problem=1
by_location=2
re_instance=3
nsamples=100
humaneval=0

if [ $data -eq $humaneval ]; then
    dataset=humaneval
    data_name=HumanEval.jsonl.gz
else
    dataset=fixeval
    data_name=fixevalA_large_problems.jsonl.gz
fi

problem_file=../data/"$dataset"/problems/"$data_name"
comp_file=../results/completion/"$dataset"/"$model_name"/"$mode"/"$fixer"_completion_"$nsamples"_"$range".jsonl
result_file="$comp_file"_results.jsonl
log_file=../results/evals/"$dataset"-"$model_name"-"$mode".txt

if [ $group -eq $by_instance ]; then
    if test -f "$comp_file"; then
        echo $dataset $mode $model_name
        python evaluator.py --sample_file $comp_file --problem_file $problem_file --log_file $log_file 
    fi
else
    if test -f "$result_file"; then
        echo $dataset $mode $model_name
        if [ $group -eq $by_problem ]; then
            python evaluator_problem.py --result_file $result_file --log_file $log_file --group problem
        fi
    else
        echo "N/A"
    fi
fi