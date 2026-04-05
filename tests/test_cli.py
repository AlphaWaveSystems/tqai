"""Smoke tests for the tqai CLI (no model downloads required)."""

from __future__ import annotations

import subprocess
import sys


def _cli(*args):
    """Run tqai CLI and return (returncode, stdout, stderr)."""
    result = subprocess.run(
        [sys.executable, "-m", "tqai", *args],
        capture_output=True,
        text=True,
    )
    return result.returncode, result.stdout, result.stderr


def test_info_exits_zero():
    code, out, _ = _cli("info")
    assert code == 0


def test_info_shows_version():
    import tqai
    _, out, _ = _cli("info")
    assert tqai.__version__ in out


def test_info_shows_backend():
    _, out, _ = _cli("info")
    assert "backend" in out.lower()


def test_benchmark_exits_zero():
    code, out, _ = _cli("benchmark", "--head-dim", "64", "--n-vectors", "50")
    assert code == 0


def test_benchmark_shows_compression_ratio():
    _, out, _ = _cli("benchmark", "--head-dim", "64", "--n-vectors", "50")
    assert "Compression" in out


def test_benchmark_asymmetric_bits():
    code, out, _ = _cli("benchmark", "--bits-k", "3", "--bits-v", "2",
                         "--head-dim", "128", "--n-vectors", "50")
    assert code == 0
    assert "Keys (3-bit)" in out
    assert "Values (2-bit)" in out


def test_no_subcommand_prints_help():
    code, out, err = _cli()
    # argparse may write to stdout or stderr depending on version
    combined = out + err
    assert "usage" in combined.lower() or "tqai" in combined.lower()


def test_unknown_subcommand_nonzero():
    code, _, _ = _cli("notacommand")
    assert code != 0
