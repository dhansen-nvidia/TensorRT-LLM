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

import io

from scripts.extract_allreduce_autotuner_timings import parse_timings, render_markdown, write_csv

_RUNNER = (
    "AllReduceRunner(tp_size=8, group=[0, 1, 2, 3, 4, 5, 6, 7], "
    "input_dtype=torch.bfloat16, fusion=NONE, input_uses_nccl_window=False)"
)


def _profile_line(tactic: int, time_ms: float, tokens: int = 16) -> str:
    return (
        f"[TensorRT-LLM] [INFO] [Autotuner] Profiled runner={_RUNNER}, "
        f"tactic={tactic}, shapes=[torch.Size([{tokens}, 4096]), torch.Size([0])]: {time_ms:.6f}ms.\n"
    )


def test_parse_and_render_allreduce_timing_table():
    timings = parse_timings(
        [
            "unrelated log line\n",
            _profile_line(0, 0.31),
            _profile_line(8, 0.29),
            _profile_line(10, 0.20),
            _profile_line(11, 0.24),
        ]
    )

    assert [timing.strategy for timing in timings] == [
        "NCCL",
        "NCCL_SYMMETRIC",
        "NCCL_RING",
        "NCCL_SYMK",
    ]
    assert all(timing.message_bytes == 128 * 1024 for timing in timings)

    table = render_markdown(timings)
    assert "| Tokens | Input shape | Message size |" in table
    assert "| 16 | `16x4096` | 128 KiB |" in table
    assert "**0.200000**" in table


def test_duplicate_timings_use_requested_aggregation():
    timings = parse_timings([_profile_line(10, 0.20), _profile_line(10, 0.40)])

    table = render_markdown(timings, aggregate="median")

    assert "**0.300000**" in table


def test_csv_has_one_column_per_strategy():
    timings = parse_timings([_profile_line(0, 0.31), _profile_line(10, 0.20)])
    output = io.StringIO()

    write_csv(timings, output)

    rows = output.getvalue().splitlines()
    assert rows[0].endswith(",NCCL,NCCL_RING")
    assert ",16,16x4096,131072,0.310000,0.200000" in rows[1]
