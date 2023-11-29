# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0


__BUGGY__ = 'buggy'
__CLEAN__ = 'clean'
__LARGEST_PROBLEM__ = 'max'

__CG_350M__ = 'codegen-350M-mono'
__CG_2B__ = 'codegen-2B-mono'
__INCODER_1B__ = 'incoder-1B'
__INCODER_6B__ = 'incoder-6B'


__REMOVAL__ = 'removal'
__REWRITING__ = 'rewriting'
__INFILL_LINE__ = 'infill_line'
__INFILL_LINE_ORACLE__ = 'infill_line_oracle'
__INFILL_SPAN__ = 'infill_span'


N_WORKERS = 96


def get_prefix(method_name):
    if method_name == 'completion':
        return '_naive'
    return method_name