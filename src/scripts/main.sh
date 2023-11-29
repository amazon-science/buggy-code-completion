# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

group=0
data=fixeval 
for model in codegen-350M-mono
do
    for method in _naive
    do
        sh scripts/eval.sh 1 0-291 clean $model $method $group
    done
done
