# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import ctypes
import os
import shutil
import subprocess
from pathlib import Path
from unittest.mock import call

import pytest

from tensorrt_llm._torch.pyexecutor import bolt_profiler as bolt_profiler_module
from tensorrt_llm._torch.pyexecutor import py_executor as py_executor_module
from tensorrt_llm._torch.pyexecutor.bolt_profiler import (
    BOLT_PROFILE_OUTPUT_DIR_ENV_VAR_NAME,
    BOLT_PROFILE_START_STOP_ENV_VAR_NAME,
    BoltProfiler,
    _BoltController,
    _BoltSymbols,
    _InstrumentedModule,
    _LoadedModule,
)
from tensorrt_llm._torch.pyexecutor.py_executor import (
    PyExecutor,
    _load_bolt_iteration_indexes,
    _load_iteration_indexes,
)


class _FakeBoltController:
    def __init__(
        self,
        modules: list[_InstrumentedModule],
        *,
        validation_error: _InstrumentedModule | None = None,
        clear_error: _InstrumentedModule | None = None,
        dump_error: _InstrumentedModule | None = None,
    ) -> None:
        self.modules = modules
        self.validation_error = validation_error
        self.clear_error = clear_error
        self.dump_error = dump_error
        self.discover_count = 0
        self.validated: list[_InstrumentedModule] = []
        self.suppressed: list[_InstrumentedModule] = []
        self.cleared: list[_InstrumentedModule] = []
        self.dumped: list[tuple[_InstrumentedModule, Path]] = []
        self.released: list[_InstrumentedModule] = []

    def discover(self) -> list[_InstrumentedModule]:
        self.discover_count += 1
        return self.modules.copy()

    def validate(self, module: _InstrumentedModule) -> None:
        self.validated.append(module)
        if module == self.validation_error:
            raise RuntimeError("validation failed")

    def suppress_final_dump(self, module: _InstrumentedModule) -> None:
        self.suppressed.append(module)

    def clear(self, module: _InstrumentedModule) -> None:
        if module == self.clear_error:
            raise RuntimeError("clear failed")
        self.cleared.append(module)

    def dump(self, module: _InstrumentedModule, output_path: Path) -> None:
        if module == self.dump_error:
            raise OSError("dump failed")
        self.dumped.append((module, output_path))

    def release(self, module: _InstrumentedModule) -> None:
        self.released.append(module)


def _make_module(
    path: Path,
    *,
    load_bias: int = 0x1000,
    device: int = 1,
    inode: int = 2,
) -> _InstrumentedModule:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.touch()
    return _InstrumentedModule(
        path=path,
        load_bias=load_bias,
        device=device,
        inode=inode,
        clear_address=load_bias + 0x10,
        dump_address=load_bias + 0x20,
        sleep_time_address=load_bias + 0x30,
    )


def _with_sleep_time(
    module: _InstrumentedModule,
    sleep_time: ctypes.c_uint32,
) -> _InstrumentedModule:
    return _InstrumentedModule(
        path=module.path,
        load_bias=module.load_bias,
        device=module.device,
        inode=module.inode,
        clear_address=module.clear_address,
        dump_address=module.dump_address,
        sleep_time_address=ctypes.addressof(sleep_time),
    )


def _make_executor() -> PyExecutor:
    executor = object.__new__(PyExecutor)
    executor.global_rank = 0
    executor.iter_counter = 0
    executor.is_warmup = False
    executor.profile_start_iters = frozenset()
    executor.profile_stop_iters = frozenset()
    executor.bolt_profile_start_iters = frozenset()
    executor.bolt_profile_stop_iters = frozenset()
    executor.print_log = False
    executor.enable_iter_perf_stats = False
    return executor


@pytest.fixture
def profiler_dependencies(mocker, monkeypatch):
    monkeypatch.delenv(py_executor_module.PROFILE_TRACE_ENV_VAR_NAME, raising=False)
    calibrator = mocker.Mock()
    mocker.patch.object(py_executor_module, "get_calibrator", return_value=calibrator)
    mocker.patch.object(py_executor_module, "get_global_profiler", return_value=None)
    mocker.patch.object(py_executor_module.torch.cuda, "Event", return_value=mocker.Mock())
    synchronize = mocker.patch.object(py_executor_module.torch.cuda, "synchronize")
    return synchronize


def test_from_env_is_disabled_without_roi(monkeypatch, mocker):
    monkeypatch.delenv(BOLT_PROFILE_START_STOP_ENV_VAR_NAME, raising=False)
    controller = mocker.patch.object(bolt_profiler_module, "_BoltController")

    assert BoltProfiler.from_env(global_rank=0) is None
    controller.assert_not_called()


def test_from_env_requires_output_directory(monkeypatch):
    monkeypatch.setenv(BOLT_PROFILE_START_STOP_ENV_VAR_NAME, "10-20")
    monkeypatch.delenv(BOLT_PROFILE_OUTPUT_DIR_ENV_VAR_NAME, raising=False)

    with pytest.raises(ValueError, match=BOLT_PROFILE_OUTPUT_DIR_ENV_VAR_NAME):
        BoltProfiler.from_env(global_rank=0)


def test_from_env_arms_loaded_modules(tmp_path, monkeypatch, mocker):
    monkeypatch.setenv(BOLT_PROFILE_START_STOP_ENV_VAR_NAME, "10-20")
    monkeypatch.setenv(BOLT_PROFILE_OUTPUT_DIR_ENV_VAR_NAME, str(tmp_path))
    arm = mocker.patch.object(BoltProfiler, "arm")

    profiler = BoltProfiler.from_env(global_rank=0)

    assert profiler is not None
    arm.assert_called_once_with()


def test_existing_profiler_singleton_behavior_is_unchanged(monkeypatch):
    env_var = "TEST_PROFILE_SINGLETON_RANGE"
    monkeypatch.setenv(env_var, "10")
    _load_iteration_indexes.cache_clear()

    assert _load_iteration_indexes(env_var) == (frozenset({10}), frozenset({10}))


def test_singleton_bolt_range_profiles_one_iteration(monkeypatch):
    env_var = "TEST_BOLT_SINGLETON_RANGE"
    monkeypatch.setenv(env_var, "10")
    _load_bolt_iteration_indexes.cache_clear()

    assert _load_bolt_iteration_indexes(env_var) == (
        frozenset({10}),
        frozenset({11}),
    )


def test_bolt_ranges_allow_adjacent_regions(monkeypatch):
    env_var = "TEST_BOLT_ADJACENT_RANGES"
    monkeypatch.setenv(env_var, "10-20, 20-30")
    _load_bolt_iteration_indexes.cache_clear()

    assert _load_bolt_iteration_indexes(env_var) == (
        frozenset({10, 20}),
        frozenset({20, 30}),
    )


@pytest.mark.parametrize(
    "value",
    (
        "10-10",
        "20-10",
        "10-20,15-25",
        "10-20,10-20",
        "-1-10",
        "10-",
        "10,,20",
    ),
)
def test_bolt_ranges_reject_invalid_configuration(monkeypatch, value):
    env_var = "TEST_BOLT_INVALID_RANGE"
    monkeypatch.setenv(env_var, value)
    _load_bolt_iteration_indexes.cache_clear()

    with pytest.raises(ValueError, match=env_var):
        _load_bolt_iteration_indexes(env_var)


def test_controller_discovers_and_pins_instrumented_modules(
    tmp_path,
    monkeypatch,
    mocker,
):
    instrumented_path = tmp_path / "libtensorrt_llm.so"
    plain_path = tmp_path / "libplain.so"
    instrumented_path.touch()
    plain_path.touch()
    instrumented = _LoadedModule(instrumented_path, 0x1000, 1, 2)
    plain = _LoadedModule(plain_path, 0x2000, 1, 3)
    monkeypatch.setattr(
        bolt_profiler_module,
        "_iter_loaded_modules",
        lambda: [plain, instrumented],
    )

    def inspect(module):
        if module != instrumented:
            return None
        return _BoltSymbols(0x10, 0x20, 0x30), os.open(instrumented_path, os.O_RDONLY)

    monkeypatch.setattr(bolt_profiler_module, "_inspect_loaded_module", inspect)
    controller = _BoltController()
    pin = mocker.patch.object(controller._loader, "pin", return_value=123)
    release = mocker.patch.object(controller._loader, "release")

    modules = controller.discover()

    assert len(modules) == 1
    assert modules[0].path == instrumented_path
    assert modules[0].clear_address == 0x1010
    assert modules[0].dump_address == 0x1020
    assert modules[0].sleep_time_address == 0x1030
    pin.assert_called_once_with(instrumented)

    controller.release(modules[0])
    release.assert_called_once_with(123)


def test_loaded_module_enumeration_is_deduplicated():
    modules = bolt_profiler_module._iter_loaded_modules()
    keys = [module.key for module in modules]

    assert modules
    assert len(keys) == len(set(keys))


def test_controller_suppresses_final_dump_and_clears(tmp_path, mocker):
    sleep_time = ctypes.c_uint32(0)
    module = _with_sleep_time(
        _make_module(tmp_path / "libtensorrt_llm.so"),
        sleep_time,
    )
    clear = mocker.Mock()
    mocker.patch.object(bolt_profiler_module, "_ClearFunction", return_value=clear)
    controller = _BoltController()

    controller.validate(module)
    controller.suppress_final_dump(module)
    controller.validate(module)
    controller.clear(module)

    assert sleep_time.value == bolt_profiler_module._FINAL_DUMP_SUPPRESSED
    clear.assert_called_once_with()


def test_controller_rejects_periodic_bolt_profile_writing(tmp_path, mocker):
    sleep_time = ctypes.c_uint32(10)
    module = _with_sleep_time(
        _make_module(tmp_path / "libtensorrt_llm.so"),
        sleep_time,
    )
    clear_function = mocker.patch.object(bolt_profiler_module, "_ClearFunction")

    with pytest.raises(RuntimeError, match="periodic profile writing enabled"):
        _BoltController().validate(module)

    clear_function.assert_not_called()


def test_profiler_arm_suppresses_without_clearing(tmp_path):
    module = _make_module(tmp_path / "libtensorrt_llm.so")
    controller = _FakeBoltController([module])
    profiler = BoltProfiler(
        global_rank=0,
        output_dir=tmp_path,
        controller=controller,
    )

    profiler.arm()

    assert controller.validated == [module]
    assert controller.suppressed == [module]
    assert controller.cleared == []
    assert controller.released == [module]


def test_profiler_rescans_and_prepares_each_region(tmp_path):
    module = _make_module(tmp_path / "libtensorrt_llm.so")
    controller = _FakeBoltController([module])
    profiler = BoltProfiler(
        global_rank=0,
        output_dir=tmp_path,
        controller=controller,
        pid=123,
    )

    profiler.start(iteration=10)
    profiler.stop(iteration=20)
    profiler.start(iteration=50)
    profiler.stop(iteration=60)

    assert controller.discover_count == 4
    assert controller.cleared == [module, module]
    assert len(controller.dumped) == 2
    assert controller.released == [module, module, module, module]
    output_names = [output_path.name for _, output_path in controller.dumped]
    assert ".roi-start-10.stop-20." in output_names[0]
    assert ".roi-start-50.stop-60." in output_names[1]


def test_profiler_validates_all_modules_before_mutating_any(tmp_path):
    first = _make_module(tmp_path / "first.so", inode=10)
    second = _make_module(tmp_path / "second.so", inode=11)
    controller = _FakeBoltController([first, second], validation_error=second)
    profiler = BoltProfiler(
        global_rank=0,
        output_dir=tmp_path,
        controller=controller,
    )

    with pytest.raises(RuntimeError, match="validate all BOLT modules"):
        profiler.start(iteration=10)

    assert controller.validated == [first, second]
    assert controller.suppressed == []
    assert controller.cleared == []
    assert controller.released == [first, second]


def test_profiler_cleans_up_successfully_cleared_modules_after_clear_failure(tmp_path):
    first = _make_module(tmp_path / "first.so", inode=10)
    second = _make_module(tmp_path / "second.so", inode=11)
    controller = _FakeBoltController([first, second], clear_error=second)
    profiler = BoltProfiler(
        global_rank=0,
        output_dir=tmp_path,
        controller=controller,
    )

    with pytest.raises(RuntimeError, match="clear counters"):
        profiler.start(iteration=10)

    assert controller.cleared == [first]
    assert len(controller.dumped) == 1
    assert ".partial-stop-10." in controller.dumped[0][1].name
    assert controller.released == [second, first, first, second]


def test_profiler_attempts_every_dump_before_raising(tmp_path):
    first = _make_module(tmp_path / "first.so", inode=10)
    second = _make_module(tmp_path / "second.so", inode=11)
    controller = _FakeBoltController([first, second], dump_error=first)
    profiler = BoltProfiler(
        global_rank=0,
        output_dir=tmp_path,
        controller=controller,
    )
    profiler.start(iteration=10)

    with pytest.raises(RuntimeError, match="complete profile stop"):
        profiler.stop(iteration=20)

    assert [module for module, _ in controller.dumped] == [second]
    assert controller.released == [first, second, first, second]
    with pytest.raises(RuntimeError, match="not active"):
        profiler.stop(iteration=20)


def test_profiler_fails_when_no_instrumented_modules_are_loaded(tmp_path):
    profiler = BoltProfiler(
        global_rank=0,
        output_dir=tmp_path,
        controller=_FakeBoltController([]),
    )

    with pytest.raises(RuntimeError, match="complete runtime control set"):
        profiler.start(iteration=10)


def test_output_names_are_unique_and_merge_compatible(tmp_path):
    first = _make_module(tmp_path / "first" / "bindings.cpython-312.so", inode=10)
    second = _make_module(tmp_path / "second" / "bindings.cpython-312.so", inode=11)
    controller = _FakeBoltController([first, second])
    profiler = BoltProfiler(
        global_rank=3,
        output_dir=tmp_path,
        controller=controller,
        pid=8123,
    )

    profiler.start(iteration=10)
    profiler.stop(iteration=17, partial=True)

    output_names = [output_path.name for _, output_path in controller.dumped]
    assert len(set(output_names)) == 2
    for output_name in output_names:
        assert output_name.startswith(
            "bindings.cpython-312.rank-3.pid-8123.roi-start-10.partial-stop-17.path-"
        )
        assert output_name.endswith(".fdata")
        assert Path(output_name).match("bindings.cpython-312.*.fdata")


def _compile_hidden_runtime(tmp_path: Path, source: str, name: str) -> Path:
    compiler = shutil.which("cc")
    if compiler is None:
        pytest.skip("A host C compiler is required for the BOLT ELF discovery test")

    source_path = tmp_path / f"{name}.c"
    library_path = tmp_path / f"{name}.so"
    source_path.write_text(source)
    subprocess.run(
        [
            compiler,
            "-shared",
            "-fPIC",
            "-fvisibility=hidden",
            str(source_path),
            "-o",
            str(library_path),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    return library_path


def test_real_hidden_symbol_discovery_clear_and_dump(tmp_path):
    library_path = _compile_hidden_runtime(
        tmp_path,
        """
        #include <stdint.h>
        #include <unistd.h>

        __attribute__((visibility("hidden")))
        uint32_t __bolt_instr_sleep_time = 0;
        static uint32_t clear_count = 0;

        __attribute__((visibility("hidden")))
        void __bolt_instr_clear_counters(void) { ++clear_count; }

        __attribute__((visibility("hidden")))
        void __bolt_instr_data_dump(
            int fd, const char *path, const uint8_t *contents, uint64_t size) {
            (void)path;
            (void)contents;
            (void)size;
            const char profile[] = "profile\\n";
            write(fd, profile, sizeof(profile) - 1);
        }

        __attribute__((visibility("default")))
        uint32_t bolt_test_clear_count(void) { return clear_count; }

        __attribute__((visibility("default")))
        uint32_t bolt_test_sleep_time(void) { return __bolt_instr_sleep_time; }
        """,
        "libbolt_test_runtime",
    )
    library = ctypes.CDLL(str(library_path), mode=os.RTLD_NOW | os.RTLD_LOCAL)
    library.bolt_test_clear_count.restype = ctypes.c_uint32
    library.bolt_test_sleep_time.restype = ctypes.c_uint32
    assert (_read_symbols := bolt_profiler_module._read_bolt_symbol_values(library_path))
    assert _read_symbols.clear_value > 0

    controller = _BoltController()
    modules = controller.discover()
    module = next(module for module in modules if module.path == library_path)
    output_path = tmp_path / "profile.fdata"
    try:
        controller.validate(module)
        controller.suppress_final_dump(module)
        controller.clear(module)
        controller.dump(module, output_path)
    finally:
        for discovered_module in modules:
            controller.release(discovered_module)

    assert library.bolt_test_sleep_time() == bolt_profiler_module._FINAL_DUMP_SUPPRESSED
    assert library.bolt_test_clear_count() == 1
    assert output_path.read_bytes() == b"profile\n"


def test_undefined_control_functions_are_rejected(tmp_path):
    library_path = _compile_hidden_runtime(
        tmp_path,
        """
        #include <stdint.h>

        extern void __bolt_instr_clear_counters(void);
        extern void __bolt_instr_data_dump(
            int fd, const char *path, const uint8_t *contents, uint64_t size);
        __attribute__((visibility("hidden")))
        uint32_t __bolt_instr_sleep_time = 0;

        __attribute__((visibility("default")))
        void bolt_test_reference_symbols(void) {
            __bolt_instr_clear_counters();
            __bolt_instr_data_dump(-1, 0, 0, 0);
        }
        """,
        "libbolt_test_undefined",
    )

    with pytest.raises(RuntimeError, match="no unique defined"):
        bolt_profiler_module._read_bolt_symbol_values(library_path)


def test_control_function_in_non_executable_section_is_rejected(tmp_path):
    library_path = _compile_hidden_runtime(
        tmp_path,
        """
        #include <stdint.h>

        __asm__(
            ".data\\n"
            ".hidden __bolt_instr_clear_counters\\n"
            ".global __bolt_instr_clear_counters\\n"
            "__bolt_instr_clear_counters:\\n"
            ".long 0\\n"
            ".text\\n");

        __attribute__((visibility("hidden")))
        uint32_t __bolt_instr_sleep_time = 0;

        __attribute__((visibility("hidden")))
        void __bolt_instr_data_dump(
            int fd, const char *path, const uint8_t *contents, uint64_t size) {
            (void)fd;
            (void)path;
            (void)contents;
            (void)size;
        }
        """,
        "libbolt_test_non_executable",
    )

    with pytest.raises(RuntimeError, match="without the required ELF permissions"):
        bolt_profiler_module._read_bolt_symbol_values(library_path)


def test_stripped_instrumented_module_has_actionable_diagnostic(tmp_path, caplog):
    strip = shutil.which("strip")
    if strip is None:
        pytest.skip("strip is required for the stripped BOLT ELF diagnostic test")
    library_path = _compile_hidden_runtime(
        tmp_path,
        """
        #include <stdint.h>

        __attribute__((visibility("hidden"), section(".bolt.instr.counters")))
        uint32_t __bolt_instr_sleep_time = 0;
        __attribute__((used, section(".bolt.instr.tables")))
        const uint8_t bolt_tables[] = {0};

        __attribute__((visibility("hidden")))
        void __bolt_instr_clear_counters(void) {}

        __attribute__((visibility("hidden")))
        void __bolt_instr_data_dump(
            int fd, const char *path, const uint8_t *contents, uint64_t size) {
            (void)fd;
            (void)path;
            (void)contents;
            (void)size;
        }
        """,
        "libbolt_test_stripped",
    )
    subprocess.run([strip, "--strip-all", str(library_path)], check=True)

    assert bolt_profiler_module._read_bolt_symbol_values(library_path) is None
    assert "Retain .symtab" in caplog.text


def test_loaded_module_inode_change_is_rejected(tmp_path):
    library_path = _compile_hidden_runtime(
        tmp_path,
        """
        #include <stdint.h>
        __attribute__((visibility("hidden")))
        uint32_t __bolt_instr_sleep_time = 0;
        __attribute__((visibility("hidden")))
        void __bolt_instr_clear_counters(void) {}
        __attribute__((visibility("hidden")))
        void __bolt_instr_data_dump(
            int fd, const char *path, const uint8_t *contents, uint64_t size) {
            (void)fd;
            (void)path;
            (void)contents;
            (void)size;
        }
        """,
        "libbolt_test_inode",
    )
    stat_result = library_path.stat()
    loaded_module = _LoadedModule(
        path=library_path,
        load_bias=0,
        device=stat_result.st_dev,
        inode=stat_result.st_ino + 1,
    )

    assert bolt_profiler_module._inspect_loaded_module(loaded_module) is None


def test_executor_profiles_multiple_bolt_regions(mocker, profiler_dependencies):
    profiler = mocker.Mock()
    mocker.patch.object(py_executor_module.BoltProfiler, "from_env", return_value=profiler)
    executor = _make_executor()
    executor.bolt_profile_start_iters = frozenset({10, 50})
    executor.bolt_profile_stop_iters = frozenset({20, 60})

    with executor._profiler() as profile_step:
        for iteration in (10, 20, 50, 60):
            executor.iter_counter = iteration
            profile_step()

    assert profiler.method_calls == [
        call.start(iteration=10),
        call.stop(iteration=20),
        call.start(iteration=50),
        call.stop(iteration=60),
    ]
    assert profiler_dependencies.call_count == 4


def test_executor_does_not_start_bolt_during_warmup(mocker, profiler_dependencies):
    profiler = mocker.Mock()
    mocker.patch.object(py_executor_module.BoltProfiler, "from_env", return_value=profiler)
    executor = _make_executor()
    executor.bolt_profile_start_iters = frozenset({10})
    executor.bolt_profile_stop_iters = frozenset({20})
    executor.iter_counter = 10
    executor.is_warmup = True

    with executor._profiler() as profile_step:
        profile_step()

    profiler.assert_not_called()
    profiler_dependencies.assert_not_called()


def test_executor_dumps_active_region_as_partial(mocker, profiler_dependencies):
    profiler = mocker.Mock()
    mocker.patch.object(py_executor_module.BoltProfiler, "from_env", return_value=profiler)
    executor = _make_executor()
    executor.bolt_profile_start_iters = frozenset({10})
    executor.bolt_profile_stop_iters = frozenset({20})

    with executor._profiler() as profile_step:
        executor.iter_counter = 10
        profile_step()
        executor.iter_counter = 17

    assert profiler.method_calls == [
        call.start(iteration=10),
        call.stop(iteration=17, partial=True),
    ]
    assert profiler_dependencies.call_count == 2


def test_executor_preserves_original_exception_when_partial_dump_fails(
    mocker,
    profiler_dependencies,
):
    profiler = mocker.Mock()
    profiler.stop.side_effect = RuntimeError("dump failed")
    mocker.patch.object(py_executor_module.BoltProfiler, "from_env", return_value=profiler)
    executor = _make_executor()
    executor.bolt_profile_start_iters = frozenset({10})

    with pytest.raises(ValueError, match="executor failed"):
        with executor._profiler() as profile_step:
            executor.iter_counter = 10
            profile_step()
            raise ValueError("executor failed")


def test_executor_propagates_partial_dump_failure_on_clean_exit(
    mocker,
    profiler_dependencies,
):
    profiler = mocker.Mock()
    profiler.stop.side_effect = RuntimeError("dump failed")
    mocker.patch.object(py_executor_module.BoltProfiler, "from_env", return_value=profiler)
    executor = _make_executor()
    executor.bolt_profile_start_iters = frozenset({10})

    with pytest.raises(RuntimeError, match="dump failed"):
        with executor._profiler() as profile_step:
            executor.iter_counter = 10
            profile_step()
