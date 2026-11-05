# SPDX-License-Identifier: GPL-2.0-or-later
"""Repairs a release shader cache lineage (user\\shader-cache) that the launcher's session race of
2026-10-19 left inconsistent (Claude, 2026-10-20). Nothing is deleted: every file taken out is
MOVED under <trash>, keeping its path.

python scripts/cache_repair.py <shader-cache dir> <trash dir> [--dry-run]

1. Undoes the unverified session (current\\.unverified + rollback\\start.txt), as the launcher does
   when the session after it does not play: the files it created go, the files the game rewrites
   (profile.bin, .idx, .vkcache) come back from rollback\\files. That session recreated stage
   metadata the race had deleted under the same names with other contents, and pipeline keys of
   the earlier sessions that use those names crashed the driver at the preload (0xC0000409).
2. Takes out the pipeline keys whose stage metadata no longer exists (the race deleted 275): a later
   session could make a stage of another specialization under such a name, and the old key would
   pair with it again."""
import argparse
import os
from pathlib import Path
import shutil
import struct


def created_100ns(path):
    # Windows creation time as a FILETIME (100 ns since 1601), what the launcher compares.
    return int(os.stat(path).st_ctime_ns // 100) + 116444736000000000


def move(src, root, trash, dry, log):
    dst = trash / src.relative_to(root)
    log.append(f'{src.relative_to(root)}')
    if not dry:
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(src), str(dst))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('lineage', type=Path)
    ap.add_argument('trash', type=Path)
    ap.add_argument('--dry-run', action='store_true')
    args = ap.parse_args()
    root = args.lineage.resolve()
    trash = args.trash.resolve()
    dry = args.dry_run
    current = root / 'current'
    cache = current / 'CUSA00004'
    rollback = root / 'rollback'
    log = []
    undone = restored = dangling = 0
    marker = current / '.unverified'
    start_file = rollback / 'start.txt'
    if marker.is_file() and start_file.is_file():
        start = int(start_file.read_text().strip())
        for p in sorted(current.rglob('*')):
            if p.is_file() and p.name != '.unverified' and created_100ns(p) >= start:
                move(p, root, trash, dry, log)
                undone += 1
        files = rollback / 'files'
        for src in sorted(files.rglob('*')) if files.is_dir() else []:
            if not src.is_file():
                continue
            dst = current / src.relative_to(files)
            if dst.exists():
                move(dst, root, trash, dry, log)
            if not dry:
                dst.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(src, dst)
            restored += 1
        move(marker, root, trash, dry, log)
        move(rollback, root, trash, dry, log)
    names = set(os.listdir(cache)) if cache.is_dir() else set()
    for name in sorted(names):
        if not name.endswith('.key') or not name.startswith(('g_', 'c_')):
            continue
        data = (cache / name).read_bytes()
        if len(data) < 16:
            continue
        stages = [struct.unpack_from('<Q', data, 8 + s * 8)[0] for s in range(6)] if name.startswith('g_') and len(data) >= 56 \
            else [struct.unpack_from('<Q', data, 8)[0]]
        if any(h and f'0x{h:016x}.meta' not in names for h in stages):
            move(cache / name, root, trash, dry, log)
            dangling += 1
    if not dry:
        trash.mkdir(parents=True, exist_ok=True)
        (trash / 'movidos.txt').write_text('\n'.join(log) + '\n', encoding='utf-8')
    print(f'{"(dry run) " if dry else ""}undone={undone} restored={restored} dangling_keys={dangling} moved_total={len(log)}')


if __name__ == '__main__':
    main()
