# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
# Modifications Copyright 2021 from HumanEval code https://github.com/openai/human-eval


from typing import List
import numpy as np
import fire
import sys, os
from utils.helpers import load_results, logging, try_mkdir
from metrics.ops import estimate_pass_at_k

# Modified from HumanEval code
def entry_point(
    result_file: str,
    ks: List[int] = [1, 10, 100],
    group: str = 'problem',
    log_file: str = None
):
    if os.path.isfile(result_file):
        samples = load_results(result_file)
        results = {}
        dct = {}

        for s in samples:
            instance_id = s[f'instance_id']
            if instance_id not in results:
                results[instance_id] = []
            results[instance_id].append(s["passed"])
            task_id = s['task_id']
            if instance_id not in dct:
                dct[instance_id] = task_id
        
        problem_dct = {}
        for instance_id, task_id in dct.items():
            if task_id not in problem_dct:
                problem_dct[task_id] = []
            problem_dct[task_id].append(instance_id)


        ans = {}
        for k in ks:
            ans[k] = []

        for task_id, instance_ids in problem_dct.items():
            total, correct = [], []
            for instance_id in instance_ids:
                passed = results[instance_id]
                total.append(len(passed))
                correct.append(sum(passed))
            total = np.array(total)
            correct = np.array(correct)
            for k in ks:
                if (total >= k).all():
                    score = estimate_pass_at_k(total, correct, k).mean()
                    ans[k].append(score)
        
        pass_at_k = {f"pass@{k}": np.asarray(ans[k]).mean() for k in ks}
        s = ''
        for v in list(pass_at_k.values()):
            s += "{:.1f} ".format(v*100)

        if log_file is not None:
            folder = os.path.dirname(log_file)
            try_mkdir(folder)
            logf = open(log_file, "a")
            prefix = os.path.basename(result_file).split('.')[0] 
            logging(logf, f"[{group}-based] {prefix}: " +  s)
            logf.close()
        else:
            print("Result: ", s)
    else:
        print('No file exists!!')

def main():
    fire.Fire(entry_point)


sys.exit(main())
