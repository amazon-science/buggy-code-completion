# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
# Modifications Copyright 2021 from HumanEval code https://github.com/openai/human-eval


import fire
import sys, os
from metrics.evaluation import evaluate_functional_correctness
from utils.helpers import logging, try_mkdir
from configs import N_WORKERS

# Modified from HumanEval code
def entry_point(
    sample_file: str,
    k: str = "1,10,100",
    n_workers: int = 4,
    timeout: float = 3.0,
    problem_file: str = '',
    n_problems = -1,
    log_file: str = None
):
    k = list(map(int, k.split(",")))
    results = evaluate_functional_correctness(sample_file, k, N_WORKERS, timeout, problem_file, n_problems)
    s = ''
    for v in list(results.values()):
        s += "{:.1f} ".format(v*100)
    if log_file is not None:
        folder = os.path.dirname(log_file)
        try_mkdir(folder)
        logf  = open(log_file, "a")
        prefix = os.path.basename(sample_file).split('.')[0] 
        logging(logf, f"{prefix}: " +  s)
        logf.close()
    else:
        print("Result: ", s)


def main():
    fire.Fire(entry_point)


sys.exit(main())
