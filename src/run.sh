# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

#!/bin/bash

WANDB_DIR='wandb'

for i in $(seq 0 1 <N>)
do
    CUDA_VISIBLE_DEVICES=$i wandb agent <> &
done

