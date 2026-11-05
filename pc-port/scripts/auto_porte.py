"""Scripted run of the source build with the user's play configuration (Claude, 2026-10-18 night).

python scripts/auto_porte.py <label> [--build DIR] [--seconds 130] [--save city|<scene>|DIR]
                             [--pad run|turn|none] [--cold] [--set K=V ...]

- Flags: exactly what Jogar-Porte.cmd (play_porte.py) gives the game, plus the pad script.
- Saves: a scene copy (run_probe isolates it; the user's saves are never written).
- Shader cache: a private copy of the user's accumulated cache, never promoted back. --cold adds a
  random ISS_SHADER_SALT and no library: every shader is new, as for a release player.
- Results: frames summary, spike classes and the run dir in out/experimental/<EXP>/auto/<label>.txt."""
import argparse
import json
import os
from pathlib import Path
import random
import shutil
import subprocess
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
import play_porte  # noqa: E402  (sets the build and the flags of the user's sessions)
play = play_porte.play
import run_probe  # noqa: E402

ROOT = play.ROOT
OUT = play.OUT
EXP = OUT / 'experimental/20261016-native-passes-v1'
SCENES = OUT / 'user/scenes'
SCRATCH = Path(os.environ.get('ISS_AUTO_SCRATCH', str(Path(os.environ.get('LOCALAPPDATA', str(OUT))) / 'iss-port/auto-cache')))


def summary(run_dir):
    import csv
    rows = list(csv.DictReader(open(run_dir / 'frames.csv')))
    f = lambda r, k: float(r.get(k) or 0)
    t = 0.0
    for r in rows:
        t += f(r, 'present_ms'); r['t'] = t / 1000
    # gameplay: from the first frame with >2000 draws plus 3 s
    start = next((r['t'] for r in rows if f(r, 'draws') > 2000), 0) + 3
    g = [r for r in rows if r['t'] > start]
    lines = [f'frames {len(rows)} session {t/1000:.1f} s, gameplay from {start:.1f} s']
    if not g:
        return '\n'.join(lines)
    tot = sum(f(r, 'present_ms') for r in g)
    pm = sorted(f(r, 'present_ms') for r in g)
    lines.append(f'gameplay {tot/1000:.1f} s  fps {len(g)*1000/tot:.1f}  p50 {pm[len(pm)//2]:.2f}  '
                 f'p99 {pm[int(len(pm)*.99)]:.2f}  max {pm[-1]:.1f}')
    lines.append('  '.join(f'>{th}ms {sum(1 for x in pm if x > th)}' for th in (25, 33, 50, 100)))
    cats = {}
    for r in g:
        p = f(r, 'present_ms')
        if p <= 25:
            continue
        if int(f(r, 'pipelines_created')) > 0:
            c = 'compile'
        elif f(r, 'readback_stall_ms') > 0.5 * f(r, 'command_processor_ms'):
            c = 'readback'
        elif int(f(r, 'image_upload_bytes')) >> 20 >= 16:
            c = 'upload16'
        else:
            c = 'cp_other'
        d = cats.setdefault(c, [0, 0.0, 0.0]); d[0] += 1; d[1] += p - 13.3; d[2] = max(d[2], p)
    for c, (n, ex, mx) in sorted(cats.items()):
        lines.append(f'  {c:9} n={n:4} excess_ms={ex:7.0f} max={mx:6.1f}')
    s = [r for r in g if f(r, 'present_ms') <= 25] or g
    keys = ['present_ms', 'command_processor_ms', 'readback_stall_ms', 'readback_async_wait_ms',
            'recorder_sync_ms', 'page_protect_ms', 'draws']
    lines.append('steady: ' + '  '.join(f'{k.replace("_ms", "")}={sum(f(r, k) for r in s)/len(s):.2f}' for k in keys))
    # 10 s buckets
    b = {}
    for r in g:
        b.setdefault(int(r['t'] // 10), []).append(r)
    lines.append('   t  |  FPS  | max ms | >33 | CP ms')
    for k, v in sorted(b.items()):
        ms = sorted(f(r, 'present_ms') for r in v)
        lines.append(f'{k*10:4d}s | {len(ms)/(sum(ms)/1000):5.1f} | {ms[-1]:6.1f} | {sum(x > 33.3 for x in ms):3d} | '
                     f'{sum(f(r, "command_processor_ms") for r in v)/len(v):5.1f}')
    return '\n'.join(lines)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('label')
    ap.add_argument('--build', type=Path, default=play.BUILD)
    ap.add_argument('--seconds', type=int, default=130)
    ap.add_argument('--save', default='city')
    ap.add_argument('--pad', default='run')
    ap.add_argument('--pad-delay', default='50')
    ap.add_argument('--cold', action='store_true')
    ap.add_argument('--resume', help='continue the private cache of an earlier label (its salt too)')
    ap.add_argument('--set', action='append', default=[])
    ap.add_argument('--cache', type=Path, help='copy this shader cache (a folder with CUSA00004) instead of the user cache')
    args = ap.parse_args()
    build = args.build if args.build.is_absolute() else (ROOT / args.build)
    exe = build / 'iss-probe.exe'
    if not exe.is_file() or not (build / 'iss-gpu.dll').is_file():
        sys.exit(f'missing build {build}')
    tasks = subprocess.check_output(['tasklist', '/FI', 'IMAGENAME eq iss-probe.exe', '/FO', 'CSV'], text=True)
    if 'iss-probe.exe' in tasks.lower():
        sys.exit('the game is already running')
    if args.save == 'city':
        saves = play.SAVES
    elif (SCENES / args.save).is_dir():
        saves = SCENES / args.save
    else:
        saves = Path(args.save)

    flags = json.loads(play.BASELINE.read_text(encoding='utf-8'))['flags']
    for key in ('ISS_PROBE_EXE', 'ISS_SAVE_ROOT', 'ISS_SAVE_SEED', 'ISS_PIPELINE_CACHE_ROOT',
                'ISS_PIPELINE_CACHE_SEED'):
        flags.pop(key, None)
    flags.update(play.NATIVE)
    flags.update(ISS_PAD_SCRIPT=args.pad, ISS_PAD_SCRIPT_DELAY=args.pad_delay, ISS_PAD_REAL='0')
    for item in args.set:
        k, sep, v = item.partition('=')
        if not sep:
            ap.error('expected KEY=VALUE')
        flags[k] = v

    SCRATCH.mkdir(parents=True, exist_ok=True)
    work = SCRATCH / args.label
    if work.exists():
        sys.exit(f'label already used: {work}')
    lib = None
    if args.resume:
        # The cache an earlier run left (a cold one keeps its salt: the same "new player").
        shutil.copytree(SCRATCH / args.resume, work)
        if (work / 'salt.txt').is_file():
            flags['ISS_SHADER_SALT'] = (work / 'salt.txt').read_text().strip()
        else:
            lib = play.library()
    elif args.cold:
        work.mkdir()
        (work / 'CUSA00004').mkdir()
        shutil.copy2(play.SEED / 'CUSA00004/profile.bin', work / 'CUSA00004/profile.bin')
        flags['ISS_SHADER_SALT'] = '%08x' % random.getrandbits(32)
        (work / 'salt.txt').write_text(flags['ISS_SHADER_SALT'])
    else:
        lib = play.library()
        shutil.copytree(args.cache or play.LINEAGE / 'current', work)
        (work / '.unverified').unlink(missing_ok=True)

    env = {k: v for k, v in os.environ.items() if not k.startswith(('ISS_', 'SHADPS4_', 'BB_'))}
    run_probe.apply_approved_environment(env, OUT)
    env.update(flags)
    env.update(ISS_PROBE_EXE=str(exe), ISS_SAVE_SEED=str(saves), ISS_PIPELINE_CACHE_ROOT=str(work),
               ISS_MINIMAL_LOGS='0')
    if lib and not any(item.startswith('ISS_PIPELINE_LIBRARY=') for item in args.set):
        env['ISS_PIPELINE_LIBRARY'] = str(lib)  # a --set library (recipes, a merged one) wins
    os.environ.clear()
    os.environ.update(env)
    print(f'auto run {args.label}: build {build.name}, save {saves.name}, pad {args.pad}, '
          f'{args.seconds} s{", cold" if args.cold else ""}', flush=True)
    shots_dir = build / 'user/gpu-2'  # the renderer's own captures (ISS_SHOT_AT, ISS_CAPTURE_FRAMES)
    shots_before = set(shots_dir.glob('*.png')) if shots_dir.is_dir() else set()
    try:
        run_probe.run(entry=True, linked=True, gpu=True, timeout=args.seconds, isolate_saves=True)
    except Exception as error:
        print('run ended with error:', error)
    record = json.loads((OUT / 'latest-run.json').read_text(encoding='utf-8'))
    run_dir = Path(record['run_dir'])
    (run_dir / 'play-flags.json').write_text(json.dumps(flags, indent=2), encoding='utf-8')
    new_shots = sorted(set(shots_dir.glob('*.png')) - shots_before) if shots_dir.is_dir() else []
    if new_shots:
        (run_dir / 'shots').mkdir(exist_ok=True)
        for i, shot in enumerate(new_shots):
            shutil.move(str(shot), str(run_dir / 'shots' / f'{i:02d}-{shot.name}'))
    text =f'== {args.label}  run: {run_dir}\n'
    stderr = (run_dir / 'stderr.log').read_text(encoding='utf-8', errors='ignore')
    if 'device lost' in stderr.lower() or (run_dir / 'gpu-crash.json').is_file():
        text += 'RESULT: GPU CRASH\n'
    try:
        text += summary(run_dir)
    except Exception as error:
        text += f'summary failed: {error}'
    dest = EXP / 'auto'
    dest.mkdir(exist_ok=True)
    (dest / f'{args.label}.txt').write_text(text + '\n', encoding='utf-8')
    print(text)


if __name__ == '__main__':
    main()
