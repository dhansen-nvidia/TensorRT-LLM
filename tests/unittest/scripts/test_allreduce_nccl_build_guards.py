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

"""CPU-only checks of the configured AllReduce build/runtime guards."""

import ast
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

_ROOT = Path(__file__).resolve().parents[3]


class TestNcclBuildGuards(unittest.TestCase):
    def test_tactic_filter(self):
        source = (_ROOT / "tensorrt_llm/_torch/custom_ops/torch_custom_ops.py").read_text()
        runner = next(
            node
            for node in ast.parse(source).body
            if isinstance(node, ast.ClassDef) and node.name == "AllReduceRunner"
        )
        method = next(
            node for node in runner.body if getattr(node, "name", "") == "get_valid_tactics"
        )
        # Execute the actual filter without importing Torch/native GPU libraries.
        namespace = {}
        exec("from __future__ import annotations\n" + ast.unparse(method), namespace)
        strategy = SimpleNamespace(
            NCCL_SYMMETRIC=SimpleNamespace(value=8),
            NCCL_RING=SimpleNamespace(value=10),
            NCCL=SimpleNamespace(value=0),
            ONESHOT=SimpleNamespace(value=4),
            TWOSHOT=SimpleNamespace(value=5),
        )
        namespace["AllReduceStrategy"] = strategy
        namespace["CustomAllReduceHelper"] = SimpleNamespace(
            max_workspace_size_auto=lambda *args, **kwargs: 100
        )
        for supported in (False, True):
            namespace["torch"] = SimpleNamespace(
                ops=SimpleNamespace(
                    trtllm=SimpleNamespace(is_nccl_allreduce_config_supported=lambda: supported)
                )
            )
            for size in (1, 1000):
                tensor = SimpleNamespace(numel=lambda: size, element_size=lambda: 2, shape=(8, 128))
                tactics = namespace["get_valid_tactics"](SimpleNamespace(tp_size=8), [tensor], None)
                self.assertEqual(10 in tactics, supported)
                self.assertIn(8, tactics)
                self.assertNotIn(11, tactics)

    @unittest.skipUnless(shutil.which("g++"), "g++ is required")
    def test_native_guards(self):
        source = (_ROOT / "cpp/tensorrt_llm/thop/allreduceOp.cpp").read_text()
        capability = source.split("bool isNcclAllReduceConfigSupported()", 1)[1].split(
            "#if ENABLE_MULTI_DEVICE\n\nnamespace", 1
        )[0]
        launch = source.split("void launchNcclAllReduce(", 1)[1].split(
            "using tensorrt_llm::common::NvmlManager;", 1
        )[0]
        prelude = """
#include <stdexcept>
#include <cstddef>
#include <cstdlib>
#define NCCL_VERSION(a,b,c) ((a)*10000+(b)*100+(c))
#define TORCH_CHECK(condition, ...) do { if (!(condition)) throw std::runtime_error("rejected"); } while (0)
#define NCCLCHECK_THROW(call) TORCH_CHECK((call) == 0)
int runtimeVersion;
int configuredCalls = 0;
int ncclGetVersion(int* version) { *version = runtimeVersion; return 0; }
using ncclDataType_t = int;
using ncclComm_t = void*;
using cudaStream_t = void*;
int ncclSum = 0;
int ncclAllReduce(void const*, void*, size_t, int, int, void*, void*) { return 0; }
#if NCCL_VERSION_CODE >= NCCL_VERSION(2,31,2)
struct ncclCollConfig_t { char const* algSelection; int forceAlgSelection; };
#define NCCL_COLLCONFIG_INITIALIZER {nullptr, 0}
int ncclAllReduceConfig(void const*, void*, size_t, int, int, void*, void*, ncclCollConfig_t* config) {
    TORCH_CHECK(config->forceAlgSelection == 1);
    ++configuredCalls;
    return 0;
}
#endif
"""
        main = """
int main(int argc, char** argv) {
    runtimeVersion = std::atoi(argv[1]);
    try {
        bool supported = isNcclAllReduceConfigSupported();
        TORCH_CHECK(supported == (NCCL_VERSION_CODE >= NCCL_VERSION(2,31,2)));
        launchNcclAllReduce(nullptr, nullptr, 1, 0, nullptr, nullptr);
        launchNcclAllReduce(nullptr, nullptr, 1, 0, nullptr, nullptr, "RING");
        return configuredCalls == 1 ? 0 : 2;
    } catch (std::runtime_error const&) {
        return configuredCalls == 0 ? 3 : 4;
    }
}
"""
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            unit = path / "guard.cpp"
            unit.write_text(
                prelude
                + "bool isNcclAllReduceConfigSupported()"
                + capability
                + "void launchNcclAllReduce("
                + launch
                + main
            )
            for headers, runtime, expected in (
                (23007, 23007, 3),
                (23102, 23102, 0),
                (23102, 23101, 3),
                (23102, 23007, 3),
            ):
                with self.subTest(headers=headers, runtime=runtime):
                    binary = path / "guard"
                    subprocess.run(
                        [
                            "g++",
                            "-std=c++17",
                            "-DENABLE_MULTI_DEVICE=1",
                            f"-DNCCL_VERSION_CODE={headers}",
                            str(unit),
                            "-o",
                            str(binary),
                        ],
                        check=True,
                        capture_output=True,
                        text=True,
                    )
                    result = subprocess.run([str(binary), str(runtime)], check=False)
                    self.assertEqual(result.returncode, expected)


if __name__ == "__main__":
    unittest.main()
