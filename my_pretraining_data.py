#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Runs create_pretraining_data.py on my data. Needed because I wanted to
1. have custom names for all files
2. merge pairs of wiki files in single TFRecord files, so that their size
   roughly equals those generated from the CC corpus
"""

from argparse import ArgumentParser
import collections
from dataclasses import dataclass
from functools import partial
from multiprocessing import Pool
import os
import re
import tqdm
from typing import List

from more_itertools import chunked, flatten


def parse_arguments():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument('--input-dir', '-i', required=True, action='append',
                        help='the input directory. Can specify more than once.')
    parser.add_argument('--output-dir', '-o', required=True,
                        help='the output directory.')
    parser.add_argument('--merge', '-m', type=int, action='append',
                        help='merge how many files. This argument should be '
                             'specified as many times as -i; if specified '
                             'less, it will be padded with 1s from the right.')
    parser.add_argument('--script-args', '-s', required=True,
                        help='the arguments to pass to '
                             'create_pretraining_data.py.')
    parser.add_argument('--processes', '-P', type=int, default=1,
                        help='number of worker processes to use (max is the '
                             'num of cores, default: 1)')
    parser.add_argument('--log-level', '-L', type=str, default='info',
                        choices=['debug', 'info', 'warning', 'error', 'critical'],
                        help='the logging level.')
    args = parser.parse_args()

    num_procs = len(os.sched_getaffinity(0))
    if args.processes < 1 or args.processes > num_procs:
        parser.error('Number of processes must be between 1 and {}'.format(
            num_procs))
    return args


@dataclass
class PretrainingRecord:
    inputs: List[str]
    output: str


def collect_input_files(input_dir, output_dir, merge=1):
    """Collects the input files -- output file records from a directory."""
    num_p = re.compile(r'(\d+)\.')
    input_files = sorted(os.listdir(input_dir))
    input_groups = list(chunked(input_files, merge))
    base_name = os.path.basename(input_dir.rstrip(os.sep))
    return [
        PretrainingRecord(
            [os.path.join(input_dir, input_file) for input_file in input_group],
            os.path.join(output_dir,
                         '_'.join([base_name] + [num_p.search(input_file).group(1)
                                                 for input_file in input_group])))
        for input_group in input_groups
    ]


def run_preprocessing(record: PretrainingRecord, script_args: str):
    os.system(f'python create_pretraining_data.py '
              f'--input_file={",".join(record.inputs)}'
              f'--output_file={record.output}.tfrecord {script_args}')


def main():
    args = parse_arguments()

    inputs = args.input_dir
    print(args.merge)
    merges = (args.merge + [1] * (len(inputs) - len(args.merge)))[:len(inputs)]

    os.nice(20)
    if not os.path.isdir(args.output_dir):
        os.makedirs(args.output_dir)

    to_process = list(flatten(
        collect_input_files(input_dir, args.output_dir, merge)
        for input_dir, merge in zip(inputs, merges))
    )
    with Pool(args.processes) as pool:
        f = partial(run_preprocessing, script_args=args.script_args)
        it = tqdm.tqdm(pool.imap_unordered(f, to_process), total=len(to_process))
        collections.deque(it, maxlen=0)


if __name__ == '__main__':
    main()
