# coding=utf-8
# Copyright 2024 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import torch
from torch.autograd import Function
from icecream import ic
#@title Math util
tiny_val = torch.finfo(torch.float32).tiny
min_val = torch.finfo(torch.float32).min
max_val = torch.finfo(torch.float32).max

def remove_zero(x):
    """Shifts `x` away from 0."""
    return torch.where(torch.abs(x) < tiny_val, tiny_val, x)

class SafeDiv(Function):
    @staticmethod
    def forward(n, d):
        r = torch.clip(n / remove_zero(d), min_val, max_val)
        return torch.where(torch.abs(d) < tiny_val, 0, r)

    @staticmethod
    def setup_context(ctx, inputs, outputs):
        n, d = inputs
        ctx.save_for_backward(n, d, outputs)

    @staticmethod
    def backward(ctx, g):
        n, d, r = ctx.saved_tensors
        dn = torch.clip(g / remove_zero(d), min_val, max_val)
        dd = torch.clip(-g * r / remove_zero(d), min_val, max_val)
        return dn, dd

class SafeSqrt(Function):
    @staticmethod
    def forward(tensor):
        mask = torch.abs(tensor) < tiny_val
        val = tensor.sqrt()
        return torch.where(mask, 0, val)

    @staticmethod
    def setup_context(ctx, inputs, outputs):
        tensor, = inputs
        ctx.save_for_backward(tensor, outputs)

    @staticmethod
    def backward(ctx, grad_output):
        tensor, val = ctx.saved_tensors
        mask = torch.abs(tensor) < tiny_val
        rcp = safe_div(1, val)
        return torch.where(
            mask, max_val * grad_output, 0.5 * rcp * grad_output)

safe_sqrt = SafeSqrt.apply
safe_div = SafeDiv.apply
