# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0


import os
from glob import glob
from typing import List, Union
import itertools
import numpy as np


normalized_input_str = '_INPUT_'

ROOT = os.path.dirname(os.path.abspath(__file__))
ATCODER_PATH = os.path.join(ROOT, "../../data/fixeval/data/atcoder_test_cases")

def read_file(infile):
    if os.path.isfile(infile):
        ins = []
        with open(infile, "r") as f:
            lines = f.readlines()
            for l in lines:
                ins.append(l.strip())
        return True, ins
    else:
        return False, None

def load_testcases(problem_test): 
    ftest = problem_test.split('atcoder_test_cases')[1]
    # tc_dir = problem_test
    tc_dir = f'{ATCODER_PATH}{ftest}'
    # print(tc_dir)
    lst_tc_files = [x.split('/')[-1] for x in glob(f"{tc_dir}/in/*")]
    testcases = {}
    for x in lst_tc_files:
        if '.in' in x:
            x = x.split('.')[0]
            x_in = x + ".in"
            x_out = x + ".out"
        else:
            x_in = x_out = x
        valid_in, ins = read_file(f"{tc_dir}/in/{x_in}")        
        valid_out, outs = read_file(f"{tc_dir}/out/{x_out}")
        if valid_in and valid_out:
            testcases[x] = {'in': ins, 'out': outs}
    # print(tc_dir, len(lst_tc_files), len(testcases))
    return testcases


def normalize(target_str, in_str, out_str, offset):
    nums = target_str.count(in_str)
    new_str = target_str
    for k in range(nums):
        new_str = new_str.replace(in_str, f"{out_str}{k+offset}_", 1)
    return new_str, nums



def format_function(func_str):
    input_strs = ['raw_input()', 'raw_input ( )', 'input()', 'input ( )', 'sys.stdin.readline()', 'stdin.readline()'] # order matters
    exit_str = 'exit()'
    exit_str_1 = 'exit ( )'
    nums = 0
    # check python 2 (raw_input) first
    new_func_str = func_str
    for _, s in enumerate(input_strs):
        new_func_str, count = normalize(new_func_str, s, normalized_input_str, nums)
        nums += count
    # exit option
    if new_func_str.count(exit_str) > 0:
        new_func_str = new_func_str.replace(exit_str, 'break')
    return new_func_str, nums


