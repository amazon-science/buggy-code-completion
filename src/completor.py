# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0


import torch, os, sys
import wandb
from tqdm import tqdm
from utils.helpers import load_results, logging, try_mkdir, separate_code, remove_lines, truncate_str
from utils.provider import load_data, load_model, write_jsonl
import configs as cfgs
import warnings
warnings.filterwarnings("ignore")

import argparse
parser = argparse.ArgumentParser(description='BCC')
parser.add_argument("-g", "--gpu_id", default=0, type=int)
parser.add_argument("--data_dir", default='data', type=str)
parser.add_argument("--log_dir", default='results/logs', type=str)
parser.add_argument("--output_dir", default='results/completion', type=str)
parser.add_argument("--use_wandb", default=0, type=int)

parser.add_argument("-m", "--mode", default='clean', type=str)
parser.add_argument("-p", "--model_name", default='codegen-350M-mono', type=str)
parser.add_argument("-d", "--dataset", default='humaneval', type=str)
parser.add_argument("--large", default=1, type=int)
parser.add_argument("--method", default='completion', type=str)
parser.add_argument("-t","--temperature", default=0.6, type=float)

parser.add_argument("--max_length", default=700, type=int) # 520
parser.add_argument("--max_to_generate", default=300, type=int)

parser.add_argument("-s", "--num_samples", default=100, type=int)
parser.add_argument("-b", "--batch_size", default=20, type=int)
parser.add_argument("-r", "--range_problems", default='', type=str)
parser.add_argument("--part_id", default=0, type=int)
parser.add_argument("--part_sz", default=-1, type=int)
parser.add_argument("--start_index", default=0, type=int)
parser.add_argument("--end_index", default=-1, type=int)

args = parser.parse_args()

max_lens = {'humaneval': {'gen': 300, 'ref': 300, 'full': 650}, 'fixeval': {'gen': 128, 'ref': 256, 'full': 600}}

max_new_tokens = max_lens[args.dataset]['ref'] if args.method == 'removal' else max_lens[args.dataset]['gen']
max_full_length = max_lens[args.dataset]['full']


###================== SETUP  ===================######

args.use_wandb = args.use_wandb == 1
if args.use_wandb:
    wandb.init(project="bugcomp-bcc")
if args.gpu_id > 0 and not args.use_wandb:
    device = torch.device(f"cuda:{args.gpu_id}")
else: 
    device = torch.device(f"cuda" if torch.cuda.is_available() else "cpu")


###================== DATA & SETTING ===================######
max_instances, problem_sets = load_data(args.data_dir, args.dataset, large=args.large == 1)

if args.end_index > -1:
    max_instances = args.end_index 

if args.part_sz == -1:
    args.part_sz = max_instances

p_start = args.part_id * args.part_sz + args.start_index
# p_end = (args.part_id + 1) * args.part_sz - 1 
p_end = p_start + args.part_sz - 1 
p_end = min(p_end, max_instances)
problems = [problem_sets[i] for i in range(p_start, p_end + 1)]

dname = f'{args.dataset}/{args.model_name}/{args.mode}'
sample_folder = f'{args.output_dir}/{dname}'
log_folder = f'{args.log_dir}/{dname}'
n_problems = len(problems)


task_id_lst, instance_id_lst, prompt_lst = [], [], []
for i, problem in enumerate(problems):
    task_id_lst.append(problem['task_id'])
    instance_id_lst.append(problem['instance_id']) #instance_id
    prompt_lst.append(remove_lines(problem[f'{args.mode}_prompt'], ''))

prefix = cfgs.get_prefix(args.method)
fname = f'{prefix}_completion_{args.num_samples}_{p_start}-{p_end}'
completion_file = f"{sample_folder}/{fname}.jsonl"
log_file = f"{log_folder}/{fname}.out"
try_mkdir(sample_folder)
try_mkdir(log_folder)
logf  = open(log_file, "w")

###================== MAIN ===================######
logging(logf, f'Number of processing problems: {n_problems} ({args.start_index} {p_start}--{p_end}) -- {args.method}')
if args.method == 'removal':
    from baselines import remove_code
    prompt_lst = [remove_code(prompt, args.dataset) for prompt in prompt_lst]
elif args.method == 'comp_fix':
    from nbfbaselines.nbfbase import NBFModel
    from baselines import fix_bugs
    fixer = NBFModel.from_pretrained("realit")
elif args.method in ['rewriting']:
    # infill partial codes
    from infiller.infill_line import localize_line
    infill_folder = f'{sample_folder}/_rewriting'
    infill_file = f"{infill_folder}/{fname}.jsonl"

    if os.path.isfile(infill_file):
        infilled_problems = load_results(infill_file)
        prompt_lst = [prob['prompt'] for prob in infilled_problems]
    else:
        try_mkdir(infill_folder)
        infiller, tokenizer = load_model('incoder-6B', device)
        # dictionary of prompts for each problem in the list
        infilled_problems = []
        lst = []
        for i, prompt in enumerate(prompt_lst):
            statement, code = separate_code(prompt)
            found_l, partial_code = localize_line(infiller, tokenizer, statement, code, 0.95, device)
            if found_l > 0:
                prompt_lst[i] = partial_code
            else:
                lst.append(i)
            infilled_problems.append(dict(id=i, task_id=task_id_lst[i], prompt=prompt_lst[i]))
        write_jsonl(infill_file, infilled_problems)
        logging(logf, 'Save infilled prompts into ' + infill_file)
        del infiller; torch.cuda.empty_cache()

###================== COMPLETION ===================######
model, tokenizer = load_model(args.model_name, device)
samples = []
fail_ids = []
try:
    for i in tqdm(range(n_problems)):
        new_samples = []
        instance_id, task_id, prompt = instance_id_lst[i], task_id_lst[i], prompt_lst[i]
        if args.use_wandb:
            wandb.log({'Processing instance': i})
        logging(logf, f"Processing instance {i} <- {instance_id}")
        try:
            completions = []
            tokens = tokenizer([prompt], return_tensors='pt') # no truncation
            input_ids=tokens['input_ids'].to(device)
            input_len = input_ids[0].flatten().size(0)
            num_generated_tokens = max_new_tokens
            batch_size = args.batch_size
            total_steps = (args.num_samples - 1)// batch_size + 1
            for i in range(total_steps):
                bz = min(batch_size, args.num_samples - batch_size * i)
                set_input_ids = input_ids.expand(bz, -1)
                with torch.no_grad():
                    if args.model_name.startswith('codegen'):
                        outs = model.generate(input_ids=set_input_ids, max_new_tokens=num_generated_tokens, pad_token_id=tokenizer.eos_token_id, do_sample=True, temperature=args.temperature)
                    else:
                        outs = model.generate(input_ids=set_input_ids, do_sample=True, top_p=0.95, temperature=0.2, max_new_tokens=num_generated_tokens)
                    outs = tokenizer.batch_decode(outs[:, input_len:], clean_up_tokenization_spaces=False, skip_special_tokens=True)
                    out_texts = [truncate_str(o) for o in outs]
                    completions += out_texts
            for c in completions:
                program = prompt + '\n' + c
                if args.method == 'comp-fix':
                    program = fix_bugs(fixer, program)
                new_samples.append(dict(instance_id=instance_id, task_id=task_id, completion=program))
        except RuntimeError as e:
            logging(logf, f"Failed instance {i}: " + str(e))
            fail_ids.append(instance_id)
            logging(logf, 'Stop because of CUDA mem ')     
            del model
            torch.cuda.empty_cache() 
            logging(logf, 'Restart model')   
            model, tokenizer = load_model(args.model_name, device)

        samples += new_samples
        write_jsonl(completion_file, new_samples, append=i>0)
    logging(logf, 'Save data into ' + completion_file)
except Exception as e: # work on python 3.x
    logging(logf, 'Failed '+ str(e))
logging(logf, 'Failed IDs: ' + ' '.join(fail_ids))
logf.close()