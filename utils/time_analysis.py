#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# this code comes from https://github.com/facebookresearch/pycls/pycls/utils/benchmark.py
"""Functions for benchmarking networks."""

# import pycls.utils.logging as lu
import torch
# from pycls.core.config import cfg
from utils.timer import Timer

WARMUP_ITER = 3
NUM_ITER = 30
BATCH_SIZE = 96
@torch.no_grad()
def compute_fw_test_time(model, inputs):
    """Computes forward test time (no grad, eval mode)."""
    # Use eval mode
    model.eval()
    # Warm up the caches
    for _cur_iter in range(WARMUP_ITER):
        model(inputs)
    # Make sure warmup kernels completed
    torch.cuda.synchronize()
    # Compute precise forward pass time
    timer = Timer()
    for _cur_iter in range(NUM_ITER):
        timer.tic()
        model(inputs)
        torch.cuda.synchronize()
        timer.toc()
    # Make sure forward kernels completed
    torch.cuda.synchronize()
    return timer.average_time


def compute_fw_bw_time(model, loss_fun, inputs, labels):
    """Computes forward backward time."""
    # Use train mode
    model.train()
    # Warm up the caches
    for _cur_iter in range(WARMUP_ITER):
        preds = model(inputs)
        loss = loss_fun(preds, labels.unsqueeze(1).float())
        loss.backward()
    # Make sure warmup kernels completed
    torch.cuda.synchronize()
    # Compute precise forward backward pass time
    fw_timer = Timer()
    bw_timer = Timer()
    for _cur_iter in range(NUM_ITER):
        # Forward
        fw_timer.tic()
        preds = model(inputs)
        loss = loss_fun(preds, labels.unsqueeze(1).float())
        torch.cuda.synchronize()
        fw_timer.toc()
        # Backward
        bw_timer.tic()
        loss.backward()
        torch.cuda.synchronize()
        bw_timer.toc()
    # Make sure forward backward kernels completed
    torch.cuda.synchronize()
    return fw_timer.average_time, bw_timer.average_time


def compute_precise_time(model, img_size, batch_size, loss_fun, device):
    """Computes precise time."""
    # img_size is a list [h, w]
    # Generate a dummy mini-batch
    inputs = torch.rand(batch_size, 1, img_size[0], img_size[1])
    labels = torch.zeros(batch_size, dtype=torch.int64)
    # Copy the data to the GPU
    # inputs = inputs.cuda(non_blocking=False)
    # labels = labels.cuda(non_blocking=False)
    inputs = inputs.to(device, non_blocking=False)
    labels = labels.to(device, non_blocking=False)
    # Compute precise time
    fw_test_time = compute_fw_test_time(model, inputs)
    # fw_time, bw_time = compute_fw_bw_time(model, loss_fun, inputs, labels)
    fw_time = 0
    bw_time = 0
    # Log precise time

    result = {
            "prec_test_fw_time": fw_test_time,
            "prec_train_fw_time": fw_time,
            "prec_train_bw_time": bw_time,
            "prec_train_fw_bw_time": fw_time + bw_time,
        }

    return result