"""Compile and run the CPU reference for the game's 48-byte feedback record."""
from __future__ import annotations

from pathlib import Path
import subprocess

from build_gpu import environment
import toolchain

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "pc-port/out/tests"


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    env = environment()
    for name in ("test_feedback_record", "test_feedback_candidates",
                 "test_feedback_identity_registry"):
        exe = OUT / f"{name}.exe"
        subprocess.run([
            str(toolchain.clang()),
            "/nologo", "/std:c++20", "/EHsc", "/MD",
            f"/I{ROOT / 'pc-port/gpu'}",
            str(ROOT / f"pc-port/tests/{name}.cpp"),
            f"/Fe:{exe}",
        ], env=env, check=True)
        subprocess.run([str(exe)], check=True)
        print(f"{name}: passed")


if __name__ == "__main__":
    main()
