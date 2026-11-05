# SPDX-License-Identifier: GPL-2.0-or-later
"""Runs a release folder's game the way its launcher does (kGameEnvironment of launcher.cpp plus
the session's settings), on COPIES of its saves and shader cache, with the window hidden
(Claude, 2026-10-20). Nothing of the release folder is written.

python scripts/release_run.py <release dir> <work dir> [--bin DIR] [--seconds 300] [--pad run|none]
                               [--set K=V ...]

Results: the run's logs in <work dir>/logs, the exit code, the frames summary (auto_porte's)."""
import argparse
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]


def game_environment():
    text = (ROOT / 'launcher/launcher.cpp').read_text(encoding='utf-8')
    block = text[text.index('kGameEnvironment[][2] = {'):]
    block = block[:block.index('};')]
    return dict(re.findall(r'\{"([A-Z0-9_]+)",\s*"([^"]*)"\}', block))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('release', type=Path)
    ap.add_argument('work', type=Path)
    ap.add_argument('--bin', type=Path, help='iss-probe.exe and iss-gpu.dll to use (default: the release bin)')
    ap.add_argument('--seconds', type=int, default=300)
    ap.add_argument('--pad', default='run')
    ap.add_argument('--set', action='append', default=[])
    ap.add_argument('--cache', type=Path, help='the shader cache (current) to copy instead of the release one')
    args = ap.parse_args()
    rel = args.release.resolve()
    work = args.work.resolve()
    if work.exists():
        sys.exit(f'work dir exists: {work}')
    ini = dict(line.split('=', 1) for line in (rel / 'user/launcher.ini').read_text(encoding='utf-8').splitlines() if '=' in line)
    bin_dir = (args.bin or rel / 'bin').resolve()
    # Copies: the release folder is never written.
    shutil.copytree(rel / 'user/saves', work / 'saves')
    shutil.copytree(args.cache or rel / 'user/shader-cache/current', work / 'cache')
    (work / 'cache/.unverified').unlink(missing_ok=True)
    (work / 'logs').mkdir(parents=True)
    env = {k: v for k, v in os.environ.items() if not k.upper().startswith(('ISS_', 'SHADPS4_', 'BB_'))}
    env.update(game_environment())
    limit = int(ini.get('fps_limit', '0') or 0)
    env.update(
        ISS_GAME_DIR=ini['game_dir'], ISS_SAVE_ROOT=str(work / 'saves'),
        ISS_PIPELINE_CACHE_ROOT=str(work / 'cache'), ISS_STOP_FILE=str(work / 'logs/stop.request'),
        ISS_FULLSCREEN='0', BB_PRESENT_MODE=ini.get('vsync', 'Immediate'), BB_FPS_LIMIT=str(limit),
        BB_FSR1=ini.get('fsr', '0'), ISS_ASYNC_SKIP_DRAWS=ini.get('shader_mode', '0'),
        ISS_GPU_BLACKBOX=ini.get('blackbox', '0'), ISS_HYBRID_BENCH_HIDDEN='1',
        ISS_PAD_SCRIPT=args.pad, ISS_PAD_SCRIPT_DELAY='50', ISS_PAD_REAL='0',
        SHADPS4_FRAME_CSV=str(work / 'logs/frames.csv'))
    if limit:
        env['BB_VBLANK_HZ'] = str(limit * ((240 + limit - 1) // limit))
    library = rel / 'shaders/recipes'
    if (library / 'manifest.json').is_file():
        env['ISS_PIPELINE_LIBRARY'] = str(library)
    if ini.get('dlc_dir'):
        env['ISS_DLC_DIR'] = ini['dlc_dir']
    for item in args.set:
        k, _, v = item.partition('=')
        env[k] = v
    image = rel / 'user/prepared/boot-linked.bin'
    out = open(work / 'logs/stdout.log', 'wb')
    err = open(work / 'logs/stderr.log', 'wb')
    start = time.time()
    p = subprocess.Popen([str(bin_dir / 'iss-probe.exe'), str(image), '--entry'], cwd=str(image.parent),
                         env=env, stdout=out, stderr=err, creationflags=subprocess.CREATE_NO_WINDOW)
    try:
        p.wait(timeout=args.seconds)
    except subprocess.TimeoutExpired:
        (work / 'logs/stop.request').write_text('stop')
        try:
            p.wait(timeout=60)
        except subprocess.TimeoutExpired:
            p.kill()
            p.wait()
    print(f'exit=0x{p.returncode & 0xFFFFFFFF:08x} after {time.time() - start:.0f} s, logs {work / "logs"}')
    frames = work / 'logs/frames.csv'
    if not frames.is_file():
        # The game writes frames.csv into the run dir of ISS_FRAME_TS / its own log dir.
        for candidate in work.rglob('frames.csv'):
            frames = candidate
            break
    if frames.is_file():
        sys.path.insert(0, str(Path(__file__).parent))
        import auto_porte
        print(auto_porte.summary(frames.parent))


if __name__ == '__main__':
    main()
