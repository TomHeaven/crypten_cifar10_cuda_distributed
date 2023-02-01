#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""
To run mpc_cifar example in distributed mode:

$ python3 examples/mpc_cifar/launcher.py \
    --evaluate \
    --model-location path-to-model/model.pth.tar \
    --resume  \
    --batch-size 1 \
    --print-freq 1 \
    --skip-plaintext \
    --distributed
"""

import argparse
import logging
import os

from distributed_launcher import DistributedLauncher


parser = argparse.ArgumentParser(description="CrypTen Cifar Training")
parser.add_argument(
    "--world_size",
    type=int,
    default=2,
    help="The number of parties to launch. Each party acts as its own process",
)
parser.add_argument(
    "--rank",
    type=int,
    help="The rank of the current party. Each party acts as its own process",
)
parser.add_argument(
    "--master_address",
    type=str,
    help="master IP Address",
)
parser.add_argument(
    "--master_port",
    type=int,
    help="master port",
)
parser.add_argument(
    "--backend",
    type=str,
    default="NCCL",
    help="backend for torhc.distributed, 'NCCL' or 'gloo'.",
)
parser.add_argument(
    "--gpu_id",
    type=int,
    default=0,
    help="which GPU to be used on the local machine. Default '0'.",
)
parser.add_argument(
    "--epochs", default=25, type=int, metavar="N", help="number of total epochs to run"
)
parser.add_argument(
    "--start-epoch",
    default=0,
    type=int,
    metavar="N",
    help="manual epoch number (useful on restarts)",
)
parser.add_argument(
    "-b",
    "--batch-size",
    default=256,
    type=int,
    metavar="N",
    help="mini-batch size (default: 256)",
)
parser.add_argument(
    "--lr",
    "--learning-rate",
    default=0.001,
    type=float,
    metavar="LR",
    help="initial learning rate",
)
parser.add_argument("--momentum", default=0.9, type=float, metavar="M", help="momentum")
parser.add_argument(
    "--weight-decay",
    "--wd",
    default=1e-6,
    type=float,
    metavar="W",
    help="weight decay (default: 1e-4)",
)
parser.add_argument(
    "--print-freq",
    "-p",
    default=10,
    type=int,
    metavar="N",
    help="print frequency (default: 10)",
)
parser.add_argument(
    "--model-location",
    default="",
    type=str,
    metavar="PATH",
    help="path to latest checkpoint (default: none)",
)
parser.add_argument(
    "--resume",
    default=False,
    action="store_true",
    help="Resume training from latest checkpoint",
)
parser.add_argument(
    "-e",
    "--evaluate",
    dest="evaluate",
    action="store_true",
    help="evaluate model on validation set",
)
parser.add_argument(
    "--seed", default=None, type=int, help="seed for initializing training. "
)
parser.add_argument("--lr-decay", default=0.1, type=float, help="lr decay factor")
parser.add_argument(
    "--skip-plaintext",
    default=False,
    action="store_true",
    help="Skip validation for plaintext network",
)
parser.add_argument(
    "--distributed",
    default=False,
    action="store_true",
    help="Run example in distributed mode",
)


def _run_experiment(args):
    # only import here to initialize crypten within the subprocesses
    from mpc_cifar import run_mpc_cifar

    # Only Rank 0 will display logs.
    level = logging.INFO
    #if "RANK" in os.environ and os.environ["RANK"] != "0":
    #    level = logging.CRITICAL
    logging.getLogger().setLevel(level)
    run_mpc_cifar(
        args.epochs,
        args.start_epoch,
        args.batch_size,
        args.lr,
        args.momentum,
        args.weight_decay,
        args.print_freq,
        args.model_location,
        args.resume,
        args.evaluate,
        args.seed,
        args.skip_plaintext,
        gpu_id=args.gpu_id
    )


def main(run_experiment):
    args = parser.parse_args()
    if args.distributed:
        launcher = DistributedLauncher(args.world_size, args.rank, args.master_address, args.master_port, 
                                       args.backend, run_experiment, args)
        launcher.start()
        #launcher.join()
        #launcher.terminate()
    else:
        run_experiment(args)


if __name__ == "__main__":
    main(_run_experiment)
