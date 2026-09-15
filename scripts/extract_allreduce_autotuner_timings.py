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

"""Extract TensorRT-LLM AllReduce autotuner timings into pivot tables."""

import argparse
import csv
import re
import statistics
import sys
from collections import defaultdict
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import TextIO

_PROFILE_PATTERN = re.compile(
    r"\[Autotuner\] Profiled runner=(?P<runner>.*?AllReduceRunner.*?)"
    r", tactic=(?P<tactic>-?\d+), shapes=\[torch\.Size\(\[(?P<dims>[^\]]*)\]\).*?"
    r": (?P<time_ms>(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?)ms\."
)
_DTYPE_PATTERN = re.compile(r"input_dtype=torch\.(?P<dtype>[A-Za-z0-9_]+)")

_TACTIC_NAMES = {
    0: "NCCL",
    1: "MIN_LATENCY",
    2: "UB",
    3: "AUTO",
    4: "ONESHOT",
    5: "TWOSHOT",
    6: "LOWPRECISION",
    7: "MNNVL",
    8: "NCCL_SYMMETRIC",
    9: "SYMM_MEM",
    10: "NCCL_RING",
    11: "NCCL_SYMK",  # Historical experiment logs; no longer an available tactic.
}
_DTYPE_BYTES = {
    "bool": 1,
    "bfloat16": 2,
    "float16": 2,
    "float32": 4,
    "float64": 8,
    "float8_e4m3fn": 1,
    "float8_e5m2": 1,
    "int8": 1,
    "int16": 2,
    "int32": 4,
    "int64": 8,
    "uint8": 1,
}


@dataclass(frozen=True)
class Timing:
    runner: str
    tactic: int
    shape: tuple[int, ...]
    time_ms: float

    @property
    def strategy(self) -> str:
        return _TACTIC_NAMES.get(self.tactic, f"TACTIC_{self.tactic}")

    @property
    def num_tokens(self) -> int:
        return self.shape[0]

    @property
    def message_bytes(self) -> int | None:
        dtype_match = _DTYPE_PATTERN.search(self.runner)
        if dtype_match is None:
            return None
        element_size = _DTYPE_BYTES.get(dtype_match.group("dtype"))
        if element_size is None:
            return None
        elements = 1
        for dimension in self.shape:
            elements *= dimension
        return elements * element_size


def parse_timings(lines: Iterable[str]) -> list[Timing]:
    timings = []
    for line in lines:
        match = _PROFILE_PATTERN.search(line)
        if match is None:
            continue
        dimensions = tuple(
            int(value.strip()) for value in match.group("dims").split(",") if value.strip()
        )
        if not dimensions:
            continue
        timings.append(
            Timing(
                runner=match.group("runner"),
                tactic=int(match.group("tactic")),
                shape=dimensions,
                time_ms=float(match.group("time_ms")),
            )
        )
    return timings


def _aggregate(values: Sequence[float], method: str) -> float:
    if method == "mean":
        return statistics.fmean(values)
    if method == "min":
        return min(values)
    if method == "max":
        return max(values)
    return statistics.median(values)


def _format_bytes(size: int | None) -> str:
    if size is None:
        return "unknown"
    units = ("B", "KiB", "MiB", "GiB", "TiB")
    value = float(size)
    for unit in units:
        if value < 1024 or unit == units[-1]:
            return f"{value:.0f} {unit}" if value.is_integer() else f"{value:.2f} {unit}"
        value /= 1024
    raise AssertionError("unreachable")


def _tables(timings: Sequence[Timing], aggregate: str):
    samples = defaultdict(list)
    for timing in timings:
        samples[(timing.runner, timing.shape, timing.strategy)].append(timing.time_ms)

    tables = defaultdict(dict)
    for (runner, shape, strategy), values in samples.items():
        tables[runner].setdefault(shape, {})[strategy] = _aggregate(values, aggregate)
    return tables


def render_markdown(timings: Sequence[Timing], aggregate: str = "median") -> str:
    sections = []
    for runner, rows in sorted(_tables(timings, aggregate).items()):
        strategies = sorted({strategy for row in rows.values() for strategy in row})
        sections.extend(
            [
                f"### {runner}",
                "",
                "| Tokens | Input shape | Message size | " + " | ".join(strategies) + " |",
                "| ---: | --- | ---: | " + " | ".join("---:" for _ in strategies) + " |",
            ]
        )
        for shape, values in sorted(rows.items(), key=lambda item: item[0]):
            known_values = list(values.values())
            winner = min(known_values) if known_values else None
            cells = []
            for strategy in strategies:
                value = values.get(strategy)
                if value is None:
                    cells.append("—")
                else:
                    rendered = f"{value:.6f}"
                    cells.append(f"**{rendered}**" if value == winner else rendered)
            representative = next(
                timing for timing in timings if timing.runner == runner and timing.shape == shape
            )
            sections.append(
                f"| {shape[0]} | `{'x'.join(map(str, shape))}` | {_format_bytes(representative.message_bytes)} | "
                + " | ".join(cells)
                + " |"
            )
        sections.append("")
    return "\n".join(sections).rstrip() + "\n"


def write_csv(timings: Sequence[Timing], output: TextIO, aggregate: str = "median") -> None:
    tables = _tables(timings, aggregate)
    strategies = sorted({timing.strategy for timing in timings})
    writer = csv.writer(output)
    writer.writerow(["configuration", "num_tokens", "input_shape", "message_bytes", *strategies])
    for runner, rows in sorted(tables.items()):
        for shape, values in sorted(rows.items(), key=lambda item: item[0]):
            representative = next(
                timing for timing in timings if timing.runner == runner and timing.shape == shape
            )
            writer.writerow(
                [
                    runner,
                    shape[0],
                    "x".join(map(str, shape)),
                    representative.message_bytes
                    if representative.message_bytes is not None
                    else "",
                    *(
                        f"{values[strategy]:.6f}" if strategy in values else ""
                        for strategy in strategies
                    ),
                ]
            )


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "log", type=Path, help="TensorRT-LLM log containing autotuner profiling lines"
    )
    parser.add_argument("--format", choices=("markdown", "csv"), default="markdown")
    parser.add_argument("--aggregate", choices=("median", "mean", "min", "max"), default="median")
    parser.add_argument("--output", type=Path, help="Write output to this file instead of stdout")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    with args.log.open("r", encoding="utf-8", errors="replace") as log_file:
        timings = parse_timings(log_file)
    if not timings:
        print(
            "No AllReduce autotuner timing lines found. Set "
            "TLLM_AUTOTUNER_LOG_LEVEL_DEBUG_TO_INFO=1 for the profiling run.",
            file=sys.stderr,
        )
        return 1

    output = args.output.open("w", encoding="utf-8", newline="") if args.output else sys.stdout
    try:
        if args.format == "csv":
            write_csv(timings, output, args.aggregate)
        else:
            output.write(render_markdown(timings, args.aggregate))
    finally:
        if args.output:
            output.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
