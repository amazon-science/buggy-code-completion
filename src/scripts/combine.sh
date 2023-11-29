# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

mode=buggy
method=removal

mode=clean
method=rewriting
data=fixeval 

echo $mode $model $method
python combine.py  --folder ../results/completion/"$data"/codegen-2B-mono/"$mode" --prefix "$method"_completion_100 --steps 0,291,37,8