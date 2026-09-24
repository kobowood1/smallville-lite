"""Runs the JavaScript unit tests (scene model of the visual town) with Node's built-in test runner."""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest

NODE = shutil.which("node")
JS_TESTS = Path(__file__).parent / "js"


@pytest.mark.skipif(NODE is None, reason="node is not installed")
def test_scene_model_js():
    files = sorted(str(p) for p in JS_TESTS.glob("*.test.mjs"))
    result = subprocess.run([NODE, "--test", *files], capture_output=True, text=True, cwd=JS_TESTS.parents[1], timeout=120)
    assert result.returncode == 0, result.stdout + result.stderr
