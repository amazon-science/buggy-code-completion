# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0


import tqdm
from nbfbaselines.nbfbase import NBFModel
from utils.helpers import load_results
from utils.provider import write_jsonl
from baselines import fix_bugs
import joblib
import argparse
parser = argparse.ArgumentParser(description='CodeLM')
parser.add_argument("-f","--fixer", default='nbf', type=str)
parser.add_argument("-i","--sample_file", default='nbf', type=str)
parser.add_argument("-o","--out_file", default='nbf', type=str)
args = parser.parse_args()
n_workers = 192

model = NBFModel.from_pretrained("realit")
programs = load_results(args.sample_file)

if '_naive' in args.sample_file:
    fixed_file = args.sample_file.replace('_naive', 'comp_fix')
else:
    fixed_file = args.out_file


def multiple_of_six(program):
    return dict(instance_id=program["instance_id"], task_id=program["task_id"], completion=fix_bugs(model, program["completion"]))

samples = joblib.Parallel(n_jobs=n_workers)(joblib.delayed(multiple_of_six)(program) for program in tqdm.tqdm(programs))

write_jsonl(fixed_file, samples)