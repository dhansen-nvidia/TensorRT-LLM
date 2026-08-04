<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# TensorRT-LLM BOLT workflow

The scripts in this directory instrument the native libraries installed with
TensorRT-LLM, collect one profile per library and process, merge those profiles
per original ELF, and use the merged data to produce BOLT-optimized libraries.
The iteration-bounded runtime controller consumes the same instrumented
libraries and the same merge pipeline; it does not run another instrumentation
pass or maintain a separate library list.

## Supported BOLT runtime

Iteration-bounded collection supports the runtime ABI emitted by LLVM BOLT
21.1.5. In particular, it depends on the hidden runtime symbols
`__bolt_instr_clear_counters`, `__bolt_instr_data_dump`, and
`__bolt_instr_sleep_time`, and on the four-argument data-dump entry point from
that release. Instrumented libraries must retain their regular ELF `.symtab`.

Use the LLVM 21.1.5 binaries staged by the internal scripts, or put the matching
`llvm-bolt` and `merge-fdata` binaries on `PATH` when using `run_local.sh`.
Using another LLVM version is unsupported for iteration-bounded collection
unless its embedded runtime ABI has been verified.

## Iteration-bounded collection

First install the libraries produced by the existing `instrument` stage. Then
configure a half-open executor iteration range and an output directory:

```bash
export TLLM_BOLT_PROFILE_START_STOP=10-20
export TLLM_BOLT_PROFILE_OUTPUT_DIR=/shared/bolt-fdata
```

This clears every discovered instrumented library immediately before executor
iteration 10 and writes its profile before iteration 20 starts. A singleton
such as `10` means the one-iteration range `[10, 11)`. Multiple non-overlapping
ranges are supported:

```bash
export TLLM_BOLT_PROFILE_START_STOP=10-20,50-60
```

Invalid, empty, reversed, negative, or overlapping ranges are rejected before
the executor loop starts. Adjacent ranges such as `10-20,20-30` are allowed.

Each file includes the library base name, global rank, PID, start and stop
iterations, and a short loaded-object identity. Early executor exit is marked
with `partial-stop`. The prefix remains compatible with the existing
`${lib_base}.*.fdata` merge glob.

When iteration-bounded profiling is configured, the controller suppresses the
embedded runtime's automatic finalization dump as soon as the executor profiler
context starts and rescans at every ROI boundary. TensorRT-LLM's instrumented
native libraries are expected to remain loaded for the executor lifetime. A
library loaded and unloaded entirely between two ROI boundaries is outside this
POC's supported lifetime model and can still execute its own finalization hook.

LLVM BOLT records native CPU control flow, not GPU kernels. The executor
synchronizes CUDA before each clear and dump to align asynchronous device work
with the iteration boundary, but auxiliary native CPU threads are not
suspended. Consequently, the boundary remains approximate.

Periodic BOLT profile writing is incompatible with this controller. Instrument
with the default `--instrumentation-sleep-time=0`; do not combine the ROI
environment variables with a positive instrumentation sleep time.

## Merge compatibility

The controller writes one `.fdata` file per library, process/rank, and ROI.
Do not combine data from different libraries manually. The existing merge stage
finds each library's files with:

```bash
${lib_base}.*.fdata
```

and invokes `merge-fdata` separately for every original ELF.
