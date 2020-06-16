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
import re
import select
import subprocess as sp
import time


def parse_arguments():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument('--command-line', '-c', required=True, metavar='FILE',
                        help='the FILE that contains the command to run '
                             'training -- without the --init_checkpoint '
                             'argument.')
    parser.add_argument('--tpu-command', '-t', required=True, metavar='FILE',
                        help='the FILE that contains the command to create '
                             'the TPU.')
    parser.add_argument('--log-file', '-l', required=True, metavar='FILE',
                        help='the log file to append to and the one that is '
                             'monitored.')
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
    delete = sp.run('yes | gcloud compute tpus delete {}'.format(tpu_name), shell=True)
    if delete.returncode:
        raise RuntimeError('Could not delete TPU {}.'.format(tpu_name))
    for i in range(3):
        create = sp.run(tpu_cmd)
        if create.returncode == 0:
            return
    else:
        raise RuntimeError('Could not create TPU {}.'.format(tpu_name), shell=True)


def last_checkpoint(output_dir):
    """
    Finds the last checkpoint in _output_dir_. Returns ``None`` if none exists.
    """
    ret = sp.run('gsutil cp {}/checkpoint .'.format(output_dir), shell=True)
    if ret.returncode == 0:
        with open('checkpoint') as inf:
            m = re.search(r'^model_checkpoint_path:\s*"([^"]+)"$', inf.read(), re.M)
        os.unlink('checkpoint')
        return os.path.join(output_dir, m.group(1))
    else:
        return None


OK, ERROR, PREEMPTED = range(3)


def run_one(full_cmd, log_file):
    train_proc = sp.Popen(full_cmd + ' >> ' + log_file + ' 2>&1', shell=True)
    # text=True, encoding='utf-8', close_fds=True)

    # Let's wait a bit so that we don't see the output from the last run
    time.sleep(5)

    tail_proc = sp.Popen('tail -f ' + log_file, shell=True, stdout=sp.PIPE,
                         bufsize=1)

    node_closed_p = re.compile('Cancelled: Node was closed')
    infeed_error_p = re.compile('ERROR:tensorflow:Error recorded from infeed')
    state_msg_p = re.compile(
        r'TPUPollingThread found TPU.*in state READY, and health (.+?)\.')

    tpu_status = OK
    timeout = 5
    old_data = b''
    try:
        while True:
            ready, _, _ = select.select([tail_proc.stdout], [], [], timeout)
            if ready:
                new_data = tail_proc.stdout.read(4096)
                # Apparently this also indicates an error (peer disconnected)
                if not new_data:
                    return tpu_status
                data = old_data + new_data
                *lines, old_data = data.split(b'\n')
                for line in lines:
                    line = line.rstrip().decode('utf-8')
                    print(line)
                    m = node_closed_p.search(line)
                    if m:
                        tpu_status = ERROR
                        logging.debug('Node was closed.')
                        continue
                    m = infeed_error_p.search(line)
                    if m:
                        tpu_status = ERROR
                        logging.debug('Infeed error.')
                        continue
                    m = state_msg_p.search(line)
                    if m:
                        if m.group(1) == 'UNHEALTHY_MAINTENANCE':
                            logging.warning('The TPU has been preempted.')
                            return PREEMPTED
                        if tpu_status == ERROR:
                            logging.warning('An error happened, and the script '
                                            'has to be restarted.')
                            return ERROR
            elif train_proc.poll() is not None:
                # Process exited
                return tpu_status
    finally:
        # Make sure the processes are stopped before returning
        if tail_proc.poll() is None:
            logging.info('Terminating tail process...')
            tail_proc.terminate()
        if train_proc.poll() is None:
            logging.info('Terminating training process...')
            train_proc.terminate()
        tail_proc.wait()
        train_proc.wait()
        logging.info('Processes terminated.')


def main():
    args = parse_arguments()
    cmd, output_dir = read_command_file(args.command_line)
    tpu_cmd = read_tpu_file(args.tpu_command)

    logging.basicConfig(
        format='%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%m/%d/%Y %H:%M:%S', level=logging.DEBUG
    )
    logging.info('Running command {}...'.format(cmd))

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
            logging.info('Found checkpoint {}.'.format(cp))
        full_cmd = cmd if not cp else '{} --init_checkpoint={}'.format(cmd, cp)

        logging.info('Full command: {}'.format(full_cmd))

        tpu_status = run_one(full_cmd, args.log_file)
        if tpu_status == OK:
            # Exit from the loop
            break

    logging.info('Done.')


if __name__ == '__main__':
    main()
