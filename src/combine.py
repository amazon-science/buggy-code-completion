# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0


from utils.helpers import load_results
from utils.provider import write_jsonl
import fire
import sys, os

def entry_point(
    folder: str = '',
    prefix: str = '_naive_completion_100',
    steps: str = "0, 99, 15, 7",
    ignore_missing: bool = False
):
    min_range, max_range, bz, num_runs = steps
    all_samples = []
    lst = []
    for i in range(num_runs):
        s = i * bz + min_range
        e = min(s + bz - 1, max_range)
        fname = f'{folder}/{prefix}_{s}-{e}.jsonl'
        if os.path.isfile(fname):
            samples = load_results(fname)
            all_samples += samples
            lst.append(len(samples))
        elif not ignore_missing:
            print(" == Missing file ", s, e)
            break
    
    if len(all_samples) < (max_range - min_range + 1) * 100:
        print(len(all_samples), (max_range - min_range + 1) * 100)
        print("Not enough samples", lst)
    else:
        completion_file = f'{folder}/{prefix}_{min_range}-{e}.jsonl'
        write_jsonl(completion_file, all_samples)


def main():
    fire.Fire(entry_point)


sys.exit(main())
