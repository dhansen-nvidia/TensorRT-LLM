# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import pytest
import torch

from tensorrt_llm._torch.autotuner import OptimizationProfile
from tensorrt_llm._torch.custom_ops.torch_custom_ops import AllReduceRunner
from tensorrt_llm.functional import AllReduceFusionOp, AllReduceStrategy


@pytest.mark.parametrize("config_supported", [False, True])
def test_allreduce_autotuner_gates_ring_on_build_support(monkeypatch, config_supported):
    monkeypatch.setattr(
        torch.ops.trtllm, "is_nccl_allreduce_config_supported", lambda: config_supported
    )
    runner = AllReduceRunner(
        tp_size=8,
        group=list(range(8)),
        input_dtype=torch.bfloat16,
        op=AllReduceFusionOp.NONE.value,
        eps=1e-6,
        trigger_completion_at_end=False,
    )

    tactics = runner.get_valid_tactics(
        [torch.empty((8, 128), dtype=torch.bfloat16)], OptimizationProfile()
    )

    expected = [
        AllReduceStrategy.NCCL_SYMMETRIC.value,
        AllReduceStrategy.NCCL.value,
    ]
    if config_supported:
        expected.insert(1, AllReduceStrategy.NCCL_RING.value)
    assert tactics[: len(expected)] == expected
    assert (AllReduceStrategy.NCCL_RING.value in tactics) == config_supported
    assert not hasattr(AllReduceStrategy, "NCCL_SYMK")
