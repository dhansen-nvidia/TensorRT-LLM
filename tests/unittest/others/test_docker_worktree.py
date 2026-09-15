# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Local CPU tests: pytest -q --confcutdir=tests/unittest/others tests/unittest/others/test_docker_worktree.py.

Exercise real Git worktrees and generated commands without Docker, Slurm or GPUs.
"""

import hashlib
import os
import re
import shutil
import subprocess
from pathlib import Path

import pytest

pytestmark = pytest.mark.cpu_only

ROOT = Path(__file__).resolve().parents[3]
JENKINS_IMAGE = "registry.example/tritondevel:shared"
JENKINS_SBSA_IMAGE = "registry.example/tritondevel:shared-sbsa"
JENKINS_ROCKY_IMAGE = "registry.example/tritondevel:shared-rocky"


def git(repo: Path, *args: str) -> str:
    return subprocess.check_output(
        ["git", "-C", str(repo), "-c", "core.hooksPath=/dev/null", *args], text=True
    ).strip()


@pytest.fixture
def checkouts(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> tuple[Path, ...]:
    monkeypatch.setenv("GIT_CONFIG_GLOBAL", os.devnull)
    monkeypatch.setenv("GIT_CONFIG_NOSYSTEM", "1")
    main = tmp_path / "main"
    main.mkdir()
    for directory in ("docker", "enroot"):
        (main / directory).mkdir()
        shutil.copyfile(ROOT / directory / "Makefile", main / directory / "Makefile")
    (main / "jenkins").mkdir()
    (main / "jenkins/current_image_tags.properties").write_text(
        "IMAGE_NAME=registry.example/internal\n"
        f"LLM_DOCKER_IMAGE={JENKINS_IMAGE}\n"
        f"LLM_SBSA_DOCKER_IMAGE={JENKINS_SBSA_IMAGE}\n"
        f"LLM_ROCKYLINUX8_PY310_DOCKER_IMAGE={JENKINS_ROCKY_IMAGE}-py310\n"
        f"LLM_ROCKYLINUX8_PY312_DOCKER_IMAGE={JENKINS_ROCKY_IMAGE}\n"
    )
    (main / "tools").mkdir()
    fake_docker = main / "tools/docker"
    fake_docker.write_text('#!/bin/sh\nprintf "%s\\n" "$*" >> "$DOCKER_LOG"\n')
    fake_docker.chmod(0o755)
    (main / "tensorrt_llm").mkdir()
    (main / "tensorrt_llm/version.py").write_text('__version__ = "1.0.0"\n')
    git(main, "init", "-q")
    git(main, "config", "user.name", "Worktree Test")
    git(main, "config", "user.email", "worktree@example.invalid")
    git(main, "add", ".")
    git(main, "commit", "-q", "-sm", "Fixture")
    # Equal basenames and detached HEADs must still yield distinct images.
    linked = []
    for parent in ("a", "b"):
        destination = tmp_path / parent / "candidate"
        destination.parent.mkdir()
        git(main, "worktree", "add", "--detach", str(destination), "HEAD")
        linked.append(destination)
    return main, *linked


def make(repo: Path, target: str, *options: str, directory: str = "docker") -> str:
    return subprocess.check_output(
        [
            "make",
            "--no-print-directory",
            "-n",
            "-C",
            str(repo / directory),
            target,
            "IS_ROOTLESS=0",
            "SH_ENV=",
            "BASH_ENV=",
            "BASE_IMAGE=base",
            "BASE_TAG=1",
            "TRITON_IMAGE=triton",
            "TRITON_BASE_TAG=1",
            "BUILD_WHEEL_ARGS=",
            "USER_NAME=developer",
            "USER_ID=1000",
            "GROUP_NAME=developer",
            "GROUP_ID=1000",
            *options,
        ],
        text=True,
    )


def execute_make(repo: Path, target: str, *options: str) -> list[str]:
    docker_log = repo.parent / f"{repo.name}-docker.log"
    docker_log.unlink(missing_ok=True)
    env = os.environ.copy()
    env["DOCKER_LOG"] = str(docker_log)
    env["PATH"] = f"{repo / 'tools'}:{env['PATH']}"
    subprocess.check_call(
        [
            "make",
            "--no-print-directory",
            "-C",
            str(repo / "docker"),
            target,
            "IS_ROOTLESS=0",
            "SH_ENV=",
            "BASH_ENV=",
            "BASE_IMAGE=base",
            "BASE_TAG=1",
            "TRITON_IMAGE=triton",
            "TRITON_BASE_TAG=1",
            "BUILD_WHEEL_ARGS=",
            "USER_NAME=developer",
            "USER_ID=1000",
            "GROUP_NAME=developer",
            "GROUP_ID=1000",
            *options,
        ],
        env=env,
    )
    lines = docker_log.read_text().splitlines()
    docker_log.unlink()
    return lines


def worktree_id(checkout: Path) -> str:
    return hashlib.sha256(str(checkout.resolve()).encode()).hexdigest()[:12]


@pytest.mark.parametrize("stage", ["devel", "release", "wheel"])
def test_local_build_run_pairs_are_distinct(checkouts: tuple[Path, ...], stage: str) -> None:
    images, containers = [], []
    for checkout in checkouts:
        build = make(checkout, f"{stage}_build", "IMAGE_TAG=experiment")
        image = re.search(r"--tag (\S+)", build).group(1)
        run = make(checkout, f"{stage}_run", "IMAGE_TAG=experiment")
        assert run.rstrip().endswith(image)
        images.append(image)
        containers.append(re.search(r"--name (\S+)", run).group(1))
    assert images[0] == f"tensorrt_llm/{stage}:experiment"
    assert len(set(images)) == len(checkouts)
    assert len(set(containers)) == len(checkouts)


def test_worktree_suffix_is_stable_path_hash(checkouts: tuple[Path, ...]) -> None:
    main, *linked = checkouts
    assert "-wt-" not in make(main, "devel_run")
    for checkout in linked:
        suffix = f"-wt-{worktree_id(checkout)}"
        output = make(checkout, "devel_run")
        assert f"tensorrt_llm/devel:latest{suffix}" in output
        assert f"--name tensorrt_llm{suffix}-devel-developer" in output


@pytest.mark.parametrize("stage", ["devel", "release", "wheel"])
def test_default_local_user_images_are_distinct(checkouts: tuple[Path, ...], stage: str) -> None:
    images = []
    for checkout in checkouts:
        output = make(checkout, f"{stage}_run", "LOCAL_USER=1")
        image = re.search(r"--tag (\S+)", output).group(1)
        assert output.rstrip().endswith(image)
        images.append(image)
    assert len(set(images)) == len(checkouts)


@pytest.mark.parametrize("target", ["ngc-devel_push", "ngc-release_push", "ngc-manifest_create"])
def test_publishing_keeps_registry_tags(checkouts: tuple[Path, ...], target: str) -> None:
    outputs = [make(checkout, target) for checkout in checkouts]
    assert all(output == outputs[0] for output in outputs)
    assert all("-wt-" not in output for output in outputs)


@pytest.mark.parametrize("stage", ["devel", "release", "wheel"])
def test_push_preserves_local_build_run_image(checkouts: tuple[Path, ...], stage: str) -> None:
    local_images = []
    for checkout in checkouts:
        output = make(checkout, f"{stage}_push", "PUSH_TO_STAGING=0")
        local = re.search(r"--tag (\S+)", output).group(1)
        assert make(checkout, f"{stage}_run").rstrip().endswith(local)
        remote = f"tensorrt_llm/{stage}:latest"
        assert f"docker push {remote};" in output
        if checkout != checkouts[0]:
            assert f"docker tag {local} {remote};" in output
        local_images.append(local)
    assert len(set(local_images)) == len(checkouts)


@pytest.mark.parametrize(
    "staging,local_user,expected_push,unexpected_push",
    [
        (
            "0",
            "0",
            "registry.example/tensorrt-llm:custom",
            "registry.example/tensorrt-llm-staging:custom",
        ),
        (
            "1",
            "0",
            "registry.example/tensorrt-llm-staging:custom",
            "registry.example/tensorrt-llm:custom",
        ),
        (
            "1",
            "1",
            "registry.example/tensorrt-llm-staging:custom-developer",
            "registry.example/tensorrt-llm:custom-developer",
        ),
    ],
)
def test_push_executes_only_selected_registry_branch(
    checkouts: tuple[Path, ...],
    staging: str,
    local_user: str,
    expected_push: str,
    unexpected_push: str,
) -> None:
    checkout = checkouts[1]
    lines = execute_make(
        checkout,
        "release_push",
        "IMAGE_WITH_TAG=registry.example/tensorrt-llm:custom",
        f"PUSH_TO_STAGING={staging}",
        f"LOCAL_USER={local_user}",
    )
    pushes = [line.removeprefix("push ") for line in lines if line.startswith("push ")]
    assert pushes == [expected_push]
    assert unexpected_push not in pushes


def test_default_local_user_push_retags_only_for_publish(
    checkouts: tuple[Path, ...],
) -> None:
    checkout = checkouts[1]
    suffix = f"-wt-{worktree_id(checkout)}"
    local = f"tensorrt_llm/release:latest{suffix}-developer"
    published = "tensorrt_llm/release:latest-developer"
    lines = execute_make(
        checkout,
        "release_push",
        "LOCAL_USER=1",
        "PUSH_TO_STAGING=0",
    )
    assert f"tag {local} {published}" in lines
    assert [line for line in lines if line.startswith("push ")] == [f"push {published}"]


def test_ngc_user_images_and_pull_run(checkouts: tuple[Path, ...]) -> None:
    user_images = []
    for checkout in checkouts:
        run = make(checkout, "ngc-devel_run", "LOCAL_USER=1", "DOCKER_PULL=1")
        assert "docker pull nvcr.io/nvidia/tensorrt-llm/devel:1.0.0\n" in run
        derived = re.search(r"--tag (\S+)", run).group(1)
        assert run.rstrip().endswith(derived)
        user_images.append(derived)
        pulled = make(checkout, "devel_run", "DOCKER_PULL=1")
        assert pulled.rstrip().endswith(make(checkout, "devel_run").split()[-1])
    assert len(set(user_images)) == len(checkouts)


@pytest.mark.parametrize("stage", ["devel", "release", "wheel"])
def test_pull_preserves_local_run_image(checkouts: tuple[Path, ...], stage: str) -> None:
    for checkout in checkouts:
        output = make(checkout, f"{stage}_pull")
        remote = f"tensorrt_llm/{stage}:latest"
        local = make(checkout, f"{stage}_run").split()[-1]
        assert f"docker pull {remote}\n" in output
        if checkout != checkouts[0]:
            assert f"docker tag {remote} {local}\n" in output


@pytest.mark.parametrize("local_user", ["0", "1"])
def test_push_explicit_image_and_user_variant(checkouts: tuple[Path, ...], local_user: str) -> None:
    image = "example.invalid/tensorrt-llm:custom"
    suffix = "-developer" if local_user == "1" else ""
    for checkout in checkouts:
        options = (
            f"IMAGE_WITH_TAG={image}",
            f"LOCAL_USER={local_user}",
            "PUSH_TO_STAGING=0",
        )
        output = make(checkout, "release_push", *options)
        local = re.findall(r"--tag (\S+)", output)[-1]
        assert local == f"{image}{suffix}"
        assert make(checkout, "release_run", *options).rstrip().endswith(local)
        assert f"docker push {image}{suffix};" in output
        explicit_suffix = make(checkout, "release_push", *options, "IMAGE_TAG_SUFFIX=-chosen")
        assert f"docker push {image}-chosen;" in explicit_suffix


def test_explicit_image_and_container_overrides(checkouts: tuple[Path, ...]) -> None:
    for checkout in checkouts:
        options = ("IMAGE_WITH_TAG=example.invalid/trtllm:custom",)
        build = make(checkout, "release_build", *options)
        local = re.search(r"--tag (\S+)", build).group(1)
        assert local == "example.invalid/trtllm:custom"
        run = make(
            checkout,
            "release_run",
            *options,
            "CONTAINER_NAME=chosen",
        )
        assert run.rstrip().endswith(local)
        assert "--name chosen-release-developer" in run


@pytest.mark.parametrize("target", ["tritondevel_build", "rockylinux8_build", "trtllm_build"])
def test_jenkins_explicit_build_tag_is_authoritative(
    checkouts: tuple[Path, ...], target: str
) -> None:
    image = "registry.example/trtllm:jenkins-build"
    for checkout in checkouts:
        output = make(checkout, target, f"IMAGE_WITH_TAG={image}")
        assert re.search(r"--tag (\S+)", output).group(1) == image


def test_jenkins_internal_release_push_uses_explicit_tag(
    checkouts: tuple[Path, ...],
) -> None:
    checkout = checkouts[1]
    image = "registry.example/release:jenkins-build"
    lines = execute_make(checkout, "trtllm_push", f"IMAGE_WITH_TAG={image}")
    build = next(line for line in lines if line.startswith("buildx build "))
    assert f"--tag {image}" in build
    assert [line for line in lines if line.startswith("push ")] == [f"push {image}"]


@pytest.mark.parametrize(
    "target,options,expected",
    [
        ("ngc-devel_run", (), "nvcr.io/nvidia/tensorrt-llm/devel:1.0.0"),
        ("ngc-release_run", (), "nvcr.io/nvidia/tensorrt-llm/release:1.0.0"),
        ("jenkins_run", (), JENKINS_IMAGE),
        ("jenkins-aarch64_run", (), JENKINS_SBSA_IMAGE),
        ("jenkins-rockylinux8_run", ("PYTHON_VERSION=3.12.3",), JENKINS_ROCKY_IMAGE),
    ],
)
def test_published_run_target_defaults_use_shared_image(
    checkouts: tuple[Path, ...], target: str, options: tuple[str, ...], expected: str
) -> None:
    containers = []
    for checkout in checkouts:
        output = make(checkout, target, *options)
        assert output.rstrip().endswith(expected)
        containers.append(re.search(r"--name (\S+)", output).group(1))
    assert len(set(containers)) == len(checkouts)


@pytest.mark.parametrize(
    "target,options",
    [
        ("ngc-devel_run", ()),
        ("jenkins_run", ()),
        ("jenkins-aarch64_run", ()),
        ("jenkins-rockylinux8_run", ("PYTHON_VERSION=3.12.3",)),
    ],
)
def test_shared_base_local_user_images_are_worktree_local(
    checkouts: tuple[Path, ...], target: str, options: tuple[str, ...]
) -> None:
    images = []
    for checkout in checkouts:
        output = make(checkout, target, *options, "LOCAL_USER=1")
        image = re.search(r"--tag (\S+)", output).group(1)
        assert output.rstrip().endswith(image)
        images.append(image)
    assert len(set(images)) == len(checkouts)


def test_git_metadata_mounts_for_docker_and_enroot(checkouts: tuple[Path, ...]) -> None:
    main, *linked = checkouts
    common = git(main, "rev-parse", "--path-format=absolute", "--git-common-dir")
    mount = f"{common}:{common}:rw"
    assert mount not in make(main, "devel_run")
    assert mount not in make(main, "run_sqsh", directory="enroot")
    assert mount not in make(main, "build_sqsh", directory="enroot")
    for checkout in linked:
        assert mount in make(checkout, "devel_run")
        assert f",{mount}" in make(checkout, "run_sqsh", directory="enroot")
        assert f",{mount}" in make(checkout, "build_sqsh", directory="enroot")


@pytest.mark.parametrize("directory,target", [("docker", "devel_run"), ("enroot", "run_sqsh")])
def test_source_paths_are_consistent_across_worktrees(
    checkouts: tuple[Path, ...], directory: str, target: str
) -> None:
    for checkout in checkouts:
        expected = "/code/tensorrt_llm"
        output = make(checkout, target, directory=directory)
        assert f"{checkout}:{expected}" in output
        assert f"--{'container-' if directory == 'enroot' else ''}workdir {expected}" in output
        if directory == "docker":
            assert f"CCACHE_DIR={expected}/cpp/.ccache" in output
            assert f"CONAN_HOME={expected}/cpp/.conan" in output
        overridden = make(checkout, target, "CODE_DIR=/custom/source", directory=directory)
        assert f"{checkout}:/custom/source" in overridden


@pytest.mark.parametrize("target", ["ngc-devel_run", "jenkins_run", "jenkins-aarch64_run"])
def test_published_run_targets_use_shared_image(checkouts: tuple[Path, ...], target: str) -> None:
    image = "nvcr.io/nvidia/tensorrt-llm/devel:1.0.0"
    containers = []
    for checkout in checkouts:
        output = make(checkout, target, f"IMAGE_WITH_TAG={image}")
        assert output.rstrip().endswith(image)
        containers.append(re.search(r"--name (\S+)", output).group(1))
        assert "--workdir /code/tensorrt_llm" in output
    assert len(set(containers)) == len(checkouts)


def test_build_guide_documents_optional_writable_git_metadata() -> None:
    guide = (ROOT / "docs/source/installation/build-from-source.md").read_text()
    section = guide.split("## Step 3: Start the Container", 1)[1]
    command = section.split("```bash\n", 1)[1].split("```", 1)[0]
    assert "--volume <path_to_tensorrt_llm_on_host>:<path_to_tensorrt_llm_in_container>" in command
    git_common_dir = "$(git rev-parse --path-format=absolute --git-common-dir)"
    assert f'--volume "{git_common_dir}:{git_common_dir}:rw"' not in command
    assert f'--volume "{git_common_dir}:{git_common_dir}:rw"' in section
    assert "--workdir <path_to_tensorrt_llm_in_container>" in command
    assert "````{admonition} git operations in a worktree" in section
    assert "git worktree move" in section


def test_container_guide_distinguishes_make_and_manual_worktree_tags() -> None:
    guide = (ROOT / "docs/source/installation/containers.md").read_text()
    make_section, manual_section = guide.split("**On systems without GNU `make`**", 1)
    assert "automatic suffix in linked worktrees" in make_section
    assert "use a distinct image tag for each worktree in both commands" in manual_section
    assert "Docker still shares identical image layers between tags" in manual_section


@pytest.mark.parametrize("directory,target", [("docker", "devel_run"), ("enroot", "run_sqsh")])
def test_source_override_controls_worktree_detection(
    checkouts: tuple[Path, ...], directory: str, target: str
) -> None:
    main, linked, _ = checkouts
    common = git(main, "rev-parse", "--path-format=absolute", "--git-common-dir")
    mount = f"{common}:{common}:rw"
    suffix = f"-wt-{worktree_id(linked)}"

    output = make(main, target, f"SOURCE_DIR={linked}", directory=directory)
    assert f"{linked}:/code/tensorrt_llm" in output
    assert mount in output
    if directory == "docker":
        assert f"tensorrt_llm/devel:latest{suffix}" in output
        assert f"--name tensorrt_llm{suffix}-devel-developer" in output

    output = make(linked, target, f"SOURCE_DIR={main}", directory=directory)
    assert f"{main}:/code/tensorrt_llm" in output
    assert mount not in output
    if directory == "docker":
        assert "tensorrt_llm/devel:latest-wt-" not in output
        assert "--name tensorrt_llm-devel-developer" in output


@pytest.mark.parametrize(
    "target,workdir",
    [("wheel_run", "/src/tensorrt_llm"), ("release_run", "/app/tensorrt_llm")],
)
def test_image_working_directories_are_preserved(
    checkouts: tuple[Path, ...], target: str, workdir: str
) -> None:
    for checkout in checkouts:
        output = make(checkout, target)
        assert f"--workdir {workdir}" in output
        output = make(checkout, target, "WORK_DIR=/custom/workdir")
        assert "--workdir /custom/workdir" in output
