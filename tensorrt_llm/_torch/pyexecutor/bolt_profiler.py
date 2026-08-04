# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Iteration-bounded control for LLVM BOLT-instrumented loaded modules.

The clear and dump boundaries are approximate because auxiliary native threads
can execute instrumented code concurrently. The executor synchronizes CUDA at
each boundary, but this controller intentionally does not suspend CPU threads.
BOLT records native CPU control flow rather than GPU kernel execution.
"""

import ctypes
import hashlib
import mmap
import os
import struct
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Optional, Protocol

from tensorrt_llm.logger import logger

BOLT_PROFILE_START_STOP_ENV_VAR_NAME = "TLLM_BOLT_PROFILE_START_STOP"
BOLT_PROFILE_OUTPUT_DIR_ENV_VAR_NAME = "TLLM_BOLT_PROFILE_OUTPUT_DIR"

_BOLT_CLEAR_SYMBOL = "__bolt_instr_clear_counters"
_BOLT_DUMP_SYMBOL = "__bolt_instr_data_dump"
_BOLT_SLEEP_TIME_SYMBOL = "__bolt_instr_sleep_time"
_BOLT_SYMBOLS = (
    _BOLT_CLEAR_SYMBOL,
    _BOLT_DUMP_SYMBOL,
    _BOLT_SLEEP_TIME_SYMBOL,
)
_BOLT_COUNTERS_SECTION = ".bolt.instr.counters"
_BOLT_TABLES_SECTION = ".bolt.instr.tables"
_FINAL_DUMP_SUPPRESSED = 0xFFFFFFFF

_ELF_HEADER = struct.Struct("<16sHHIQQQIHHHHHH")
_ELF_PROGRAM_HEADER = struct.Struct("<IIQQQQQQ")
_ELF_SECTION_HEADER = struct.Struct("<IIQQQQIIQQ")
_ELF_SYMBOL = struct.Struct("<IBBHQQ")
_ELF_MAGIC = b"\x7fELF"
_ELF_CLASS_64 = 2
_ELF_DATA_LITTLE_ENDIAN = 1
_PT_LOAD = 1
_PF_EXECUTE = 0x1
_PF_WRITE = 0x2
_PF_READ = 0x4
_SHT_SYMTAB = 2
_SHN_UNDEFINED = 0
_SHF_WRITE = 0x1
_SHF_ALLOCATE = 0x2
_SHF_EXECUTE = 0x4
_STT_NOTYPE = 0
_STT_OBJECT = 1
_STT_FUNCTION = 2


@dataclass(frozen=True)
class _ElfSection:
    name_offset: int
    section_type: int
    flags: int
    address: int
    offset: int
    size: int
    link: int
    entry_size: int
    name: str = ""


@dataclass(frozen=True)
class _ElfSegment:
    flags: int
    address: int
    size: int


@dataclass(frozen=True)
class _ElfSymbol:
    value: int
    size: int
    symbol_type: int
    section_index: int


@dataclass(frozen=True)
class _BoltSymbols:
    clear_value: int
    dump_value: int
    sleep_time_value: int


@dataclass(frozen=True)
class _LoadedModule:
    path: Path
    load_bias: int
    device: int
    inode: int
    is_main_executable: bool = False

    @property
    def key(self) -> tuple[int, int, int]:
        return self.device, self.inode, self.load_bias


@dataclass(frozen=True)
class _InstrumentedModule:
    path: Path
    load_bias: int
    device: int
    inode: int
    clear_address: int
    dump_address: int
    sleep_time_address: int
    pin_handle: Optional[int] = field(default=None, repr=False, compare=False)
    file_descriptor: Optional[int] = field(default=None, repr=False, compare=False)

    @property
    def key(self) -> tuple[int, int, int]:
        return self.device, self.inode, self.load_bias

    @property
    def library_base(self) -> str:
        name = self.path.name
        return name[:-3] if name.endswith(".so") else name

    @property
    def identity(self) -> str:
        value = f"{self.path}:{self.device}:{self.inode}:{self.load_bias}"
        return hashlib.sha256(value.encode()).hexdigest()[:12]


class _BoltControllerProtocol(Protocol):
    def discover(self) -> list[_InstrumentedModule]: ...

    def validate(self, module: _InstrumentedModule) -> None: ...

    def suppress_final_dump(self, module: _InstrumentedModule) -> None: ...

    def clear(self, module: _InstrumentedModule) -> None: ...

    def dump(self, module: _InstrumentedModule, output_path: Path) -> None: ...

    def release(self, module: _InstrumentedModule) -> None: ...


class _DlPhdrInfo(ctypes.Structure):
    _fields_ = [
        ("dlpi_addr", ctypes.c_void_p),
        ("dlpi_name", ctypes.c_char_p),
        ("dlpi_phdr", ctypes.c_void_p),
        ("dlpi_phnum", ctypes.c_ushort),
    ]


_DlIterateCallback = ctypes.CFUNCTYPE(
    ctypes.c_int,
    ctypes.POINTER(_DlPhdrInfo),
    ctypes.c_size_t,
    ctypes.c_void_p,
)
_ClearFunction = ctypes.CFUNCTYPE(None)
_DumpFunction = ctypes.CFUNCTYPE(
    None,
    ctypes.c_int,
    ctypes.c_char_p,
    ctypes.POINTER(ctypes.c_uint8),
    ctypes.c_uint64,
)


def _range_fits(contents_size: int, offset: int, size: int) -> bool:
    return offset <= contents_size and size <= contents_size - offset


def _read_c_string(contents: mmap.mmap, offset: int, limit: int) -> str:
    if offset < 0 or offset >= limit:
        return ""
    end = contents.find(b"\0", offset, limit)
    if end < 0:
        end = limit
    return contents[offset:end].decode(errors="replace")


def _read_elf_sections(
    contents: mmap.mmap,
    *,
    section_offset: int,
    section_entry_size: int,
    section_count: int,
    section_name_index: int,
) -> Optional[list[_ElfSection]]:
    if section_entry_size < _ELF_SECTION_HEADER.size or section_count == 0:
        return None
    if not _range_fits(contents.size(), section_offset, section_entry_size * section_count):
        return None

    sections = []
    for index in range(section_count):
        raw = _ELF_SECTION_HEADER.unpack_from(contents, section_offset + index * section_entry_size)
        sections.append(
            _ElfSection(
                name_offset=raw[0],
                section_type=raw[1],
                flags=raw[2],
                address=raw[3],
                offset=raw[4],
                size=raw[5],
                link=raw[6],
                entry_size=raw[9],
            )
        )

    if section_name_index >= len(sections):
        return sections
    names = sections[section_name_index]
    if not _range_fits(contents.size(), names.offset, names.size):
        return sections
    names_limit = names.offset + names.size
    return [
        _ElfSection(
            name_offset=section.name_offset,
            section_type=section.section_type,
            flags=section.flags,
            address=section.address,
            offset=section.offset,
            size=section.size,
            link=section.link,
            entry_size=section.entry_size,
            name=_read_c_string(contents, names.offset + section.name_offset, names_limit),
        )
        for section in sections
    ]


def _read_elf_segments(
    contents: mmap.mmap,
    *,
    program_offset: int,
    program_entry_size: int,
    program_count: int,
) -> list[_ElfSegment]:
    if program_entry_size < _ELF_PROGRAM_HEADER.size or program_count == 0:
        return []
    if not _range_fits(contents.size(), program_offset, program_entry_size * program_count):
        return []

    segments = []
    for index in range(program_count):
        raw = _ELF_PROGRAM_HEADER.unpack_from(contents, program_offset + index * program_entry_size)
        if raw[0] == _PT_LOAD:
            segments.append(_ElfSegment(flags=raw[1], address=raw[3], size=raw[6]))
    return segments


def _read_named_symbols(
    contents: mmap.mmap,
    sections: list[_ElfSection],
) -> dict[str, list[_ElfSymbol]]:
    symbols: dict[str, list[_ElfSymbol]] = {name: [] for name in _BOLT_SYMBOLS}
    for symbol_table in sections:
        if symbol_table.section_type != _SHT_SYMTAB or symbol_table.link >= len(sections):
            continue
        string_table = sections[symbol_table.link]
        if (
            not _range_fits(contents.size(), string_table.offset, string_table.size)
            or symbol_table.entry_size < _ELF_SYMBOL.size
            or not _range_fits(contents.size(), symbol_table.offset, symbol_table.size)
        ):
            continue

        strings_limit = string_table.offset + string_table.size
        symbols_limit = symbol_table.offset + symbol_table.size
        for offset in range(
            symbol_table.offset,
            symbols_limit - _ELF_SYMBOL.size + 1,
            symbol_table.entry_size,
        ):
            raw = _ELF_SYMBOL.unpack_from(contents, offset)
            name = _read_c_string(
                contents,
                string_table.offset + raw[0],
                strings_limit,
            )
            if name in symbols:
                symbols[name].append(
                    _ElfSymbol(
                        value=raw[4],
                        size=raw[5],
                        symbol_type=raw[1] & 0xF,
                        section_index=raw[3],
                    )
                )
    return symbols


def _validate_symbol(
    *,
    path: Path,
    name: str,
    candidates: list[_ElfSymbol],
    sections: list[_ElfSection],
    segments: list[_ElfSegment],
    allowed_types: tuple[int, ...],
    required_section_flags: int,
    required_segment_flags: int,
    minimum_size: int,
) -> _ElfSymbol:
    defined = [
        symbol
        for symbol in candidates
        if symbol.section_index != _SHN_UNDEFINED and symbol.section_index < len(sections)
    ]
    unique = {
        (symbol.value, symbol.size, symbol.symbol_type, symbol.section_index) for symbol in defined
    }
    if len(unique) != 1:
        raise RuntimeError(
            f"BOLT-instrumented module {path} has no unique defined {name} in .symtab"
        )

    symbol = defined[0]
    section = sections[symbol.section_index]
    if symbol.symbol_type not in allowed_types:
        raise RuntimeError(
            f"BOLT-instrumented module {path} has an incompatible {name} symbol type "
            f"{symbol.symbol_type}"
        )
    if section.flags & required_section_flags != required_section_flags:
        raise RuntimeError(
            f"BOLT-instrumented module {path} places {name} in section "
            f"{section.name or symbol.section_index} without the required ELF permissions"
        )
    if (
        symbol.value < section.address
        or minimum_size > section.address + section.size - symbol.value
    ):
        raise RuntimeError(f"BOLT-instrumented module {path} has an out-of-bounds {name} symbol")

    for segment in segments:
        if (
            symbol.value >= segment.address
            and minimum_size <= segment.address + segment.size - symbol.value
            and segment.flags & required_segment_flags == required_segment_flags
        ):
            return symbol
    raise RuntimeError(f"BOLT-instrumented module {path} has no compatible load segment for {name}")


def _read_bolt_symbol_values_from_contents(
    contents: mmap.mmap,
    path: Path,
) -> Optional[_BoltSymbols]:
    if len(contents) < _ELF_HEADER.size:
        return None
    header = _ELF_HEADER.unpack_from(contents)
    ident = header[0]
    if ident[:4] != _ELF_MAGIC or ident[4] != _ELF_CLASS_64 or ident[5] != _ELF_DATA_LITTLE_ENDIAN:
        return None

    sections = _read_elf_sections(
        contents,
        section_offset=header[6],
        section_entry_size=header[11],
        section_count=header[12],
        section_name_index=header[13],
    )
    if sections is None:
        return None
    segments = _read_elf_segments(
        contents,
        program_offset=header[5],
        program_entry_size=header[9],
        program_count=header[10],
    )
    if not segments:
        return None

    candidates = _read_named_symbols(contents, sections)
    if not all(candidates[name] for name in _BOLT_SYMBOLS):
        section_names = {section.name for section in sections}
        if {_BOLT_COUNTERS_SECTION, _BOLT_TABLES_SECTION} <= section_names:
            missing = ", ".join(name for name in _BOLT_SYMBOLS if not candidates[name])
            logger.warning(
                "Ignoring BOLT-instrumented module %s because its regular symbol "
                "table does not expose the complete runtime control set; missing: %s. "
                "Retain .symtab when packaging instrumented libraries.",
                path,
                missing,
            )
        return None

    clear = _validate_symbol(
        path=path,
        name=_BOLT_CLEAR_SYMBOL,
        candidates=candidates[_BOLT_CLEAR_SYMBOL],
        sections=sections,
        segments=segments,
        allowed_types=(_STT_NOTYPE, _STT_FUNCTION),
        required_section_flags=_SHF_ALLOCATE | _SHF_EXECUTE,
        required_segment_flags=_PF_READ | _PF_EXECUTE,
        minimum_size=1,
    )
    dump = _validate_symbol(
        path=path,
        name=_BOLT_DUMP_SYMBOL,
        candidates=candidates[_BOLT_DUMP_SYMBOL],
        sections=sections,
        segments=segments,
        allowed_types=(_STT_NOTYPE, _STT_FUNCTION),
        required_section_flags=_SHF_ALLOCATE | _SHF_EXECUTE,
        required_segment_flags=_PF_READ | _PF_EXECUTE,
        minimum_size=1,
    )
    sleep_time = _validate_symbol(
        path=path,
        name=_BOLT_SLEEP_TIME_SYMBOL,
        candidates=candidates[_BOLT_SLEEP_TIME_SYMBOL],
        sections=sections,
        segments=segments,
        allowed_types=(_STT_NOTYPE, _STT_OBJECT),
        required_section_flags=_SHF_ALLOCATE | _SHF_WRITE,
        required_segment_flags=_PF_READ | _PF_WRITE,
        minimum_size=ctypes.sizeof(ctypes.c_uint32),
    )
    return _BoltSymbols(
        clear_value=clear.value,
        dump_value=dump.value,
        sleep_time_value=sleep_time.value,
    )


def _read_bolt_symbol_values(path: Path) -> Optional[_BoltSymbols]:
    """Return validated BOLT runtime symbol values for an ELF module."""
    try:
        file_obj = path.open("rb")
    except OSError as error:
        logger.debug("Unable to inspect loaded module %s: %s", path, error)
        return None

    with file_obj:
        try:
            contents = mmap.mmap(file_obj.fileno(), 0, access=mmap.ACCESS_READ)
        except (OSError, ValueError) as error:
            logger.debug("Unable to map loaded module %s: %s", path, error)
            return None
        with contents:
            return _read_bolt_symbol_values_from_contents(contents, path)


def _inspect_loaded_module(
    module: _LoadedModule,
) -> Optional[tuple[_BoltSymbols, int]]:
    try:
        file_obj = module.path.open("rb")
    except OSError as error:
        logger.debug("Unable to inspect loaded module %s: %s", module.path, error)
        return None

    with file_obj:
        stat_result = os.fstat(file_obj.fileno())
        if (stat_result.st_dev, stat_result.st_ino) != (module.device, module.inode):
            logger.warning(
                "Skipping loaded module %s because its path now refers to a different file",
                module.path,
            )
            return None
        try:
            contents = mmap.mmap(file_obj.fileno(), 0, access=mmap.ACCESS_READ)
        except (OSError, ValueError) as error:
            logger.debug("Unable to map loaded module %s: %s", module.path, error)
            return None
        with contents:
            symbols = _read_bolt_symbol_values_from_contents(contents, module.path)
        if symbols is None:
            return None
        return symbols, os.dup(file_obj.fileno())


def _iter_loaded_modules() -> list[_LoadedModule]:
    libc = ctypes.CDLL(None)
    try:
        dl_iterate_phdr = libc.dl_iterate_phdr
    except AttributeError as error:
        raise RuntimeError("dl_iterate_phdr is unavailable on this platform") from error

    modules: list[_LoadedModule] = []
    seen: set[tuple[int, int, int]] = set()

    @_DlIterateCallback
    def callback(info_ptr, _size, _data):
        info = info_ptr.contents
        raw_name = info.dlpi_name
        is_main_executable = not raw_name
        path = Path(os.fsdecode(raw_name) if raw_name else "/proc/self/exe")
        try:
            resolved_path = path.resolve(strict=True)
            stat_result = resolved_path.stat()
        except OSError:
            return 0

        load_bias = int(info.dlpi_addr or 0)
        identity = (stat_result.st_dev, stat_result.st_ino, load_bias)
        if identity not in seen:
            seen.add(identity)
            modules.append(
                _LoadedModule(
                    path=resolved_path,
                    load_bias=load_bias,
                    device=stat_result.st_dev,
                    inode=stat_result.st_ino,
                    is_main_executable=is_main_executable,
                )
            )
        return 0

    dl_iterate_phdr.argtypes = [_DlIterateCallback, ctypes.c_void_p]
    dl_iterate_phdr.restype = ctypes.c_int
    result = dl_iterate_phdr(callback, None)
    if result != 0:
        raise RuntimeError(f"dl_iterate_phdr failed with status {result}")
    return modules


class _DynamicLoader:
    """Pins loaded ELF objects while raw addresses from them are in use."""

    def __init__(self) -> None:
        self._libc = ctypes.CDLL(None)
        try:
            self._dlopen = self._libc.dlopen
            self._dlclose = self._libc.dlclose
            self._dlerror = self._libc.dlerror
        except AttributeError as error:
            raise RuntimeError("dlopen/dlclose are unavailable on this platform") from error
        self._dlopen.argtypes = [ctypes.c_char_p, ctypes.c_int]
        self._dlopen.restype = ctypes.c_void_p
        self._dlclose.argtypes = [ctypes.c_void_p]
        self._dlclose.restype = ctypes.c_int
        self._dlerror.argtypes = []
        self._dlerror.restype = ctypes.c_char_p

    def pin(self, module: _LoadedModule) -> int:
        self._dlerror()
        name = None if module.is_main_executable else os.fsencode(module.path)
        flags = os.RTLD_NOW
        if not module.is_main_executable:
            flags |= os.RTLD_NOLOAD | os.RTLD_LOCAL
        handle = self._dlopen(name, flags)
        if not handle:
            raw_error = self._dlerror()
            detail = os.fsdecode(raw_error) if raw_error else "unknown dynamic loader error"
            raise RuntimeError(f"Unable to pin loaded module {module.path}: {detail}")
        return int(handle)

    def release(self, handle: int) -> None:
        if self._dlclose(handle) != 0:
            raw_error = self._dlerror()
            detail = os.fsdecode(raw_error) if raw_error else "unknown dynamic loader error"
            raise RuntimeError(f"dlclose failed: {detail}")


class _BoltController:
    """Discovers and invokes runtimes embedded in BOLT-instrumented modules."""

    def __init__(self) -> None:
        self._loader = _DynamicLoader()

    def discover(self) -> list[_InstrumentedModule]:
        candidates = []
        completed = False
        try:
            for loaded_module in _iter_loaded_modules():
                inspected = _inspect_loaded_module(loaded_module)
                if inspected is None:
                    continue
                symbols, file_descriptor = inspected
                try:
                    pin_handle = self._loader.pin(loaded_module)
                except RuntimeError as error:
                    os.close(file_descriptor)
                    logger.debug("%s", error)
                    continue
                candidates.append(
                    _InstrumentedModule(
                        path=loaded_module.path,
                        load_bias=loaded_module.load_bias,
                        device=loaded_module.device,
                        inode=loaded_module.inode,
                        clear_address=loaded_module.load_bias + symbols.clear_value,
                        dump_address=loaded_module.load_bias + symbols.dump_value,
                        sleep_time_address=loaded_module.load_bias + symbols.sleep_time_value,
                        pin_handle=pin_handle,
                        file_descriptor=file_descriptor,
                    )
                )

            live_keys = {module.key for module in _iter_loaded_modules()}
            modules = []
            for module in candidates:
                if module.key in live_keys:
                    modules.append(module)
                else:
                    self.release(module)
            completed = True
            return modules
        finally:
            if not completed:
                for module in candidates:
                    self.release(module)

    def validate(self, module: _InstrumentedModule) -> None:
        sleep_time = ctypes.c_uint32.from_address(module.sleep_time_address)
        if sleep_time.value not in (0, _FINAL_DUMP_SUPPRESSED):
            raise RuntimeError(
                f"BOLT-instrumented module {module.path} already has periodic profile "
                "writing enabled; iteration-bounded profiling requires "
                "--instrumentation-sleep-time=0"
            )

    def suppress_final_dump(self, module: _InstrumentedModule) -> None:
        # Setup has already observed zero and therefore did not fork a writer.
        # A nonzero value makes LLVM BOLT 21.1.5 skip its DT_FINI dump.
        ctypes.c_uint32.from_address(module.sleep_time_address).value = _FINAL_DUMP_SUPPRESSED

    def clear(self, module: _InstrumentedModule) -> None:
        _ClearFunction(module.clear_address)()

    def dump(self, module: _InstrumentedModule, output_path: Path) -> None:
        if module.file_descriptor is None:
            raise RuntimeError(f"BOLT module {module.path} has no pinned ELF file")

        output_flags = os.O_WRONLY | os.O_CREAT | os.O_TRUNC
        output_flags |= getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
        output_fd = os.open(output_path, output_flags, 0o600)
        try:
            with mmap.mmap(
                module.file_descriptor,
                0,
                access=mmap.ACCESS_COPY,
            ) as contents:
                contents_buffer = (ctypes.c_uint8 * len(contents)).from_buffer(contents)
                try:
                    _DumpFunction(module.dump_address)(
                        output_fd,
                        os.fsencode(module.path),
                        contents_buffer,
                        len(contents),
                    )
                finally:
                    del contents_buffer
        finally:
            os.close(output_fd)

    def release(self, module: _InstrumentedModule) -> None:
        if module.file_descriptor is not None:
            try:
                os.close(module.file_descriptor)
            except OSError as error:
                logger.warning("Unable to close pinned ELF file for %s: %s", module.path, error)
        if module.pin_handle is not None:
            try:
                self._loader.release(module.pin_handle)
            except RuntimeError as error:
                logger.warning("Unable to release loaded module %s: %s", module.path, error)


_ModuleError = tuple[_InstrumentedModule, Exception]


class BoltProfiler:
    """Controls iteration-bounded profiles in already-instrumented ELF modules."""

    def __init__(
        self,
        *,
        global_rank: int,
        output_dir: Path,
        controller: Optional[_BoltControllerProtocol] = None,
        pid: Optional[int] = None,
    ) -> None:
        self._global_rank = global_rank
        self._output_dir = output_dir
        self._controller = _BoltController() if controller is None else controller
        self._pid = os.getpid() if pid is None else pid
        self._active_modules: Optional[list[_InstrumentedModule]] = None
        self._start_iteration: Optional[int] = None

    @classmethod
    def from_env(cls, *, global_rank: int) -> Optional["BoltProfiler"]:
        if not os.environ.get(BOLT_PROFILE_START_STOP_ENV_VAR_NAME):
            return None

        output_dir_value = os.environ.get(BOLT_PROFILE_OUTPUT_DIR_ENV_VAR_NAME)
        if not output_dir_value:
            raise ValueError(
                f"{BOLT_PROFILE_OUTPUT_DIR_ENV_VAR_NAME} must be set when "
                f"{BOLT_PROFILE_START_STOP_ENV_VAR_NAME} is configured"
            )
        output_dir = Path(output_dir_value).expanduser().resolve()
        output_dir.mkdir(parents=True, exist_ok=True)
        profiler = cls(global_rank=global_rank, output_dir=output_dir)
        profiler.arm()
        return profiler

    def _apply_to_all(
        self,
        modules: list[_InstrumentedModule],
        action: Callable[[_InstrumentedModule], None],
    ) -> list[_ModuleError]:
        errors = []
        for module in modules:
            try:
                action(module)
            except Exception as error:
                errors.append((module, error))
        return errors

    @staticmethod
    def _raise_module_errors(action: str, errors: list[_ModuleError]) -> None:
        details = "; ".join(
            f"{module.path}: {type(error).__name__}: {error}" for module, error in errors
        )
        raise RuntimeError(f"Unable to {action} all BOLT modules: {details}") from errors[0][1]

    def _release_all(self, modules: list[_InstrumentedModule]) -> None:
        for module in modules:
            self._controller.release(module)

    def _validate_and_suppress(
        self,
        modules: list[_InstrumentedModule],
    ) -> None:
        errors = self._apply_to_all(modules, self._controller.validate)
        if errors:
            self._raise_module_errors("validate", errors)
        errors = self._apply_to_all(modules, self._controller.suppress_final_dump)
        if errors:
            self._raise_module_errors("suppress final dumps for", errors)

    def arm(self) -> None:
        """Suppress automatic final dumps without changing current counters."""
        modules = self._controller.discover()
        if not modules:
            logger.warning(
                "BOLT ROI profiling is configured, but no controllable instrumented "
                "modules are loaded yet; discovery will be retried at ROI start"
            )
            return
        try:
            self._validate_and_suppress(modules)
            logger.info(
                "Suppressed automatic final BOLT dumps for %d loaded module(s)",
                len(modules),
            )
        finally:
            self._release_all(modules)

    def start(self, *, iteration: int) -> None:
        if self._active_modules is not None:
            raise RuntimeError("BOLT profiling region is already active")

        modules = self._controller.discover()
        if not modules:
            raise RuntimeError(
                "BOLT ROI profiling was requested, but no loaded ELF modules expose "
                f"the complete runtime control set ({', '.join(_BOLT_SYMBOLS)}) in "
                "their regular symbol table. Ensure LLVM BOLT 21.1.5 instrumented "
                "artifacts retain .symtab."
            )

        validated = False
        try:
            self._validate_and_suppress(modules)
            validated = True
        finally:
            if not validated:
                self._release_all(modules)

        logger.info(
            "Starting BOLT profile region at iteration %d on rank %d; "
            "clearing %d instrumented module(s)",
            iteration,
            self._global_rank,
            len(modules),
        )
        prepared = []
        errors: list[_ModuleError] = []
        for module in modules:
            try:
                logger.info("Clearing BOLT counters in %s", module.path)
                self._controller.clear(module)
            except Exception as error:
                errors.append((module, error))
                self._controller.release(module)
            else:
                prepared.append(module)

        if errors:
            if prepared:
                self._active_modules = prepared
                self._start_iteration = iteration
                try:
                    self.stop(iteration=iteration, partial=True)
                except RuntimeError as error:
                    errors.append((prepared[0], error))
            self._raise_module_errors("clear counters for", errors)

        self._active_modules = prepared
        self._start_iteration = iteration

    def stop(
        self,
        *,
        iteration: int,
        partial: bool = False,
    ) -> None:
        if self._active_modules is None or self._start_iteration is None:
            raise RuntimeError("BOLT profiling region is not active")

        modules = self._active_modules
        start_iteration = self._start_iteration
        errors: list[_ModuleError] = []
        for module in modules:
            output_path = self._make_output_path(
                module=module,
                start_iteration=start_iteration,
                iteration=iteration,
                partial=partial,
            )
            try:
                self._controller.dump(module, output_path)
            except Exception as error:
                errors.append((module, error))
            else:
                logger.info("Wrote BOLT profile for %s to %s", module.path, output_path)
            finally:
                self._controller.release(module)

        self._active_modules = None
        self._start_iteration = None
        try:
            self.arm()
        except RuntimeError as error:
            errors.append((modules[0], error))
        if errors:
            self._raise_module_errors("complete profile stop for", errors)

    def _make_output_path(
        self,
        *,
        module: _InstrumentedModule,
        start_iteration: int,
        iteration: int,
        partial: bool,
    ) -> Path:
        stop_component = f"partial-stop-{iteration}" if partial else f"stop-{iteration}"
        filename = (
            f"{module.library_base}.rank-{self._global_rank}.pid-{self._pid}."
            f"roi-start-{start_iteration}.{stop_component}.path-{module.identity}.fdata"
        )
        return self._output_dir / filename
