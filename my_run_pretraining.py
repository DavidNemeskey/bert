#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Runs run_pretraining.py and checks the logs it emits to detect any problems
with the TPU. The script handles two errors:

1. When e.g. a HW error happens, run_pretraining.py is interrupted and just
   started again.
2. When the TPU goes down for maintenance or is preempted, it is deleted and
   recreated before the script is run again.

In both cases, the --init_checkpoint parameter is set to the latest checkpoint.
"""

from argparse import ArgumentParser
import logging
import os
import subprocess as sp
import re


def parse_arguments():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument('--command-line', '-c', required=True, metavar='FILE',
                        help='the FILE that contains the command to run '
                             'training -- without the --init_checkpoint '
                             'argument.')
    parser.add_argument('--tpu-command', '-t', required=True, metavar='FILE',
                        help='the FILE that contains the command to create '
                             'the TPU.')
    return parser.parse_args()


def read_command_file(cmd_file):
    """
    Reads the command line file and returns the command, as well as the output
    directory.
    """
    with open(cmd_file) as inf:
        cmd = inf.readline().strip()
    output_dir = re.search(r'--output_dir\s*=\s*([^ ]+)', cmd).group(1).rstrip('/')
    return cmd, output_dir


def read_tpu_file(tpu_file):
    """
    Reads the file that contains the TPU creation command. Returns it and the
    name of the TPU.
    """
    with open(tpu_file) as inf:
        tpu_cmd = inf.readline().strip()
    return tpu_cmd


def restart_tpu(tpu_cmd):
    """Deletes and recreates the TPU."""
    tpu_name = re.search(r'gcloud\s+compute\s+tpus\s+create\s+(\S+)', tpu_cmd).group(1)
    delete = sp.run(f'yes | gcloud compute tpus delete {tpu_name}')
    if delete.returncode:
        raise RuntimeError(f'Could not delete TPU {tpu_name}.')
    for i in range(3):
        create = sp.run(tpu_cmd)
        if create.returncode == 0:
            return
    else:
        raise RuntimeError(f'Could not create TPU {tpu_name}.')


def last_checkpoint(output_dir: str) -> str:
    """
    Finds the last checkpoint in _output_dir_. Returns ``None`` if none exists.
    """
    ret = sp.run(f'gsutil cp {output_dir}/checkpoint .', shell=True)
    if ret.returncode == 0:
        with open('checkpoint') as inf:
            m = re.search(r'^model_checkpoint_path:\s*"([^"]+)"$', inf.read(), re.M)
        os.unlink('checkpoint')
        return os.path.join(output_dir, m.group(1))
    else:
        return None


def main():
    args = parse_arguments()
    cmd, output_dir = read_command_file(args.cmd_file)
    tpu_cmd = read_tpu_file(args.tpu_command)

    logging.basicConfig(
        format='%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%m/%d/%Y %H:%M:%S', level=logging.DEBUG
    )
    logging.info(f'Running command {cmd}...')

    node_closed_p = re.compile('Cancelled: Node was closed')
    state_msg_p = re.compile(r'TPUPollingThread found TPU.*in state READY, and health (.+?)\.')

    OK, ERROR, PREEMPTED = range(3)
    tpu_status = OK

    while True:
        if tpu_status == PREEMPTED:
            try:
                restart_tpu(tpu_cmd)
            except Exception:
                logging.exception('Error restarting the TPU.')
                break
        tpu_status = OK

        cp = last_checkpoint(output_dir)
        if cp:
            logging.info(f'Found checkpoint {cp}.')
        full_cmd = cmd if not cp else f'{cmd} --init_checkpoint={cp}'

        proc = sp.Popen(full_cmd, shell=True, stdout=sp.PIPE, stderr=sp.STDOUT,
                        text=True, encoding='utf-8')

        while True:
            line = proc.stdout.readline()
            if not line:
                break
            else:
                print(line.strip())
                m = node_closed_p.search(line)
                if m:
                    tpu_status = ERROR
                    logging.debug('Node was closed.')
                    continue
                m = state_msg_p.search(line)
                if m:
                    if m.group(1) == 'UNHEALTHY_MAINTENANCE':
                        # TODO preempted
                        tpu_status = PREEMPTED
                        logging.warning('The TPU has been preempted.')
                        break
                    if tpu_status == ERROR:
                        logging.warning('An error happened, and the script '
                                        'has to be restarted.')
                        break

        # We broke from the inner loop
        if tpu_status == OK:
            # Exit from the inner loop
            break
        else:
            logging.info('Terminating process...')
            proc.terminate()

    logging.info('Done.')


if __name__ == '__main__':
    main()
