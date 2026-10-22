# SPDX-License-Identifier: GPL-2.0-or-later
"""Prepare an inFAMOUS Windows entry probe and an evidence-backed PS4 service map.

Reads only executables/modules/SFO, never game assets. No guest code is executed.
"""
import argparse
from collections import Counter
import csv
import hashlib
import json
import os
from pathlib import Path
import re
import struct
import sys

from ps4_format import parse_self, span, unpack, sfo

ROOT = Path(__file__).resolve().parents[1]
WORKSPACE = ROOT.parent
# The game folder: --game, else ISS_GAME_DIR, else the first line of out/game_dir.txt (local,
# git-ignored), else the current folder.
def _default_game():
    if os.environ.get('ISS_GAME_DIR'):
        return Path(os.environ['ISS_GAME_DIR'])
    local = ROOT / 'out' / 'game_dir.txt'
    if local.is_file():
        line = local.read_text(encoding='utf-8').strip().splitlines()
        if line:
            return Path(line[0])
    return Path('.')


DEFAULT_GAME = _default_game()
# CUSA00004 (Europe) and CUSA00223 (North America) ship this same eboot.bin.
TITLE_IDS = ('CUSA00004', 'CUSA00223')
EXPECTED_EBOOT = '2d1ca79630d7bbe6fa29575f59aa7996d43071d74ac8e4f7d40137967d9039ab'
REFERENCE_COMMIT = '4d4d20812c0af23a11c8b8d40f5e06a9054925ef'


def encode_id(value):
    alphabet = 'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+-'
    encoded = alphabet[value & 63]
    while value >> 6:
        value >>= 6
        encoded = alphabet[value & 63] + encoded
    return encoded


def module(path):
    source = path.read_bytes()
    elf, header, ph, self_segments, missing = parse_self(source)
    dyn = next(p for p in ph if p['type'] == 2)
    dynamic = []
    if dyn['filesz'] % 16:
        raise ValueError('invalid dynamic table length')
    for pos in range(dyn['offset'], dyn['offset'] + dyn['filesz'], 16):
        tag, value = unpack('<QQ', elf, pos)
        if not tag:
            break
        dynamic.append((tag, value))
    tags = dict(dynamic)
    if tags.get(0x6100003b, 24) != 24 or tags.get(0x61000033, 24) != 24:
        raise ValueError('unexpected symbol/relocation entry size')
    lib = next(p for p in ph if p['type'] == 0x61000000)
    blob = span(elf, lib['offset'], lib['filesz'])
    strings = span(blob, tags[0x61000035], tags[0x61000037])

    def string(off):
        span(strings, off, 1)
        return strings[off:strings.index(0, off)].decode('ascii')

    libraries, modules = {}, {}
    for tag, value in dynamic:
        if tag in (0x61000013, 0x61000015, 0x6100000d, 0x6100000f):
            table = libraries if tag in (0x61000013, 0x61000015) else modules
            key = encode_id(value >> 48)
            identity = {'name': string(value & 0xffffffff), 'version_raw': (value >> 32) & 65535}
            if key in table and table[key] != identity:
                raise ValueError('conflicting library/module identity')
            table[key] = identity
    symbols = []
    syms = span(blob, tags[0x61000039], tags[0x6100003f])
    if len(syms) % 24:
        raise ValueError('invalid symbol table length')
    for pos in range(0, len(syms), 24):
        name, info, other, section, value, size = unpack('<IBBHQQ', syms, pos)
        text = string(name)
        parts = text.split('#')
        identity = None
        if text and len(parts) != 1:
            if len(parts) != 3:
                raise ValueError(f'unsupported symbol encoding: {text}')
            identity = {'nid': parts[0], 'library': libraries[parts[1]], 'module': modules[parts[2]]}
        symbols.append(dict(name=text, type=info & 15, binding=info >> 4,
                            section=section, value=value, size=size, identity=identity))
    relocations = []
    for ot, st in ((0x61000029, 0x6100002d), (0x6100002f, 0x61000031)):
        table = span(blob, tags.get(ot, 0), tags.get(st, 0))
        if len(table) % 24:
            raise ValueError('invalid relocation table length')
        for target, info, addend in struct.iter_unpack('<QQq', table):
            kind, sym = info & 0xffffffff, info >> 32
            if sym >= len(symbols):
                raise ValueError('relocation references nonexistent symbol')
            relocations.append((target, kind, sym, addend))
    return dict(path=str(path.resolve()), sha256=hashlib.sha256(source).hexdigest(),
                elf_sha256=hashlib.sha256(elf).hexdigest(), elf=elf, header=header,
                ph=ph, symbols=symbols, relocs=relocations, libraries=libraries,
                modules=modules, tags=tags, needed=[string(v) for t, v in dynamic if t == 1],
                missing=missing, self_segments=self_segments)


def identity_key(symbol):
    ident = symbol['identity']
    if ident is None:
        return None
    return (ident['nid'], ident['library']['name'], ident['library']['version_raw'],
            ident['module']['name'], ident['module']['version_raw'])


def names_database():
    path = WORKSPACE / 'research/shadPS4-perf/src/core/aerolib/aerolib.inl'
    if not path.exists():
        return {}, None
    text = path.read_text(encoding='utf-8')
    pairs = re.findall(r'STUB\(\s*"([^"]+)"\s*,\s*([^\s,)]+)\s*\)', text)
    return dict(pairs), {'path': str(path), 'sha256': hashlib.sha256(path.read_bytes()).hexdigest()}


def category(library, api):
    text = library + ' ' + api
    for label, needles in (
        ('graphics', ('Gnm', 'VideoOut', 'Gnmx')),
        ('input', ('Pad', 'Move')),
        ('audio', ('Audio', 'Ajm', 'Atrac')),
        ('save', ('SaveData',)),
        ('files', ('Fios', 'KernelOpen', 'KernelRead', 'KernelLseek', 'KernelStat', 'KernelClose')),
        ('memory', ('Mmap', 'Munmap', 'Memory', 'Mprotect')),
        ('threads_sync', ('Pthread', 'Sema', 'EventFlag', 'Equeue')),
        ('time', ('Clock', 'Time', 'Sleep', 'Rtc')),
        ('libc', ('libc', 'Libc')),
    ):
        if any(needle in text for needle in needles):
            return label
    return 'other_services'


def write_probe(main, out):
    loads = [p for p in main['ph'] if p['type'] in (1, 0x61000010)]
    size = max(p['vaddr'] + p['memsz'] for p in loads)
    entry = main['header'][4]

    def mapped(address, n=8, executable=False):
        return any((not executable or p['flags'] & 1) and p['vaddr'] <= address
                   and address + n <= p['vaddr'] + p['memsz'] for p in loads)

    if not 0 < size <= 512 * 1024 * 1024 or not mapped(entry, 1, True):
        raise ValueError('invalid image size or entry point')
    image = bytearray(size)
    for p in loads:
        if p['filesz'] > p['memsz'] or p['vaddr'] % 4096 or p['flags'] > 7:
            raise ValueError('unsupported load segment')
        image[p['vaddr']:p['vaddr'] + p['filesz']] = span(main['elf'], p['offset'], p['filesz'])
    imports, relocs = {}, []
    for target, kind, sym, addend in main['relocs']:
        if not mapped(target):
            raise ValueError('relocation target outside mapped memory')
        if kind == 8:
            if not 0 <= addend < size:
                raise ValueError('relative relocation value outside image')
            relocs.append((target, 0, addend, 0))
        elif kind in (1, 6, 7):
            symbol = main['symbols'][sym]
            if symbol['section']:
                value = symbol['value'] + addend
                if symbol['section'] == 0xfff1:
                    struct.pack_into('<Q', image, target, value)
                else:
                    if not 0 <= value < size:
                        raise ValueError('defined relocation outside image')
                    relocs.append((target, 0, value, 0))
            else:
                if symbol['type'] not in (1, 2):
                    raise ValueError(f'unsupported import symbol type {symbol["type"]}')
                imports.setdefault(sym, len(imports))
                rkind = 2 if symbol['type'] == 1 else 1
                if (rkind == 1 and addend) or not 0 <= addend < 4096:
                    raise ValueError('unsupported import addend')
                relocs.append((target, rkind, imports[sym], addend))
        else:
            raise ValueError(f'unsupported relocation type {kind}; execution blocked')
    names = [main['symbols'][i]['name'] for i in imports]
    with (out / 'boot.bin').open('wb') as f:
        f.write(struct.pack('<8s5Q', b'ISSPRB01', size, entry, len(loads), len(relocs), len(names)))
        for p in loads:
            f.write(struct.pack('<3Q', p['vaddr'], p['memsz'], p['flags']))
        for name in names:
            encoded = name.encode('ascii')
            if len(encoded) >= 128:
                raise ValueError('import name too long')
            f.write(encoded.ljust(128, b'\0'))
        for relocation in relocs:
            f.write(struct.pack('<4Q', *relocation))
        f.write(image)
    return dict(size=size, entry=hex(entry), segments=len(loads),
                relocations=len(relocs), imports=names,
                sha256=hashlib.sha256((out / 'boot.bin').read_bytes()).hexdigest())


def prepare(game, out):
    # Always isolate generated files from the read-only source and old emulation work.
    out = out.resolve()
    out.relative_to(ROOT.resolve())
    if out == ROOT.resolve():
        raise ValueError('use an output subdirectory inside pc-port')
    if out == game.resolve() or game.resolve() in out.parents:
        raise ValueError('output must not be inside the game directory')
    main = module(game / 'eboot.bin')
    if main['sha256'] != EXPECTED_EBOOT:
        raise ValueError('unsupported eboot identity; review a new build before executing')
    meta = sfo((game / 'sce_sys/param.sfo').read_bytes())
    if meta.get('TITLE_ID') not in TITLE_IDS:
        raise ValueError('unexpected title ID')
    bundled = [module(p) for p in sorted((game / 'sce_module').glob('*.prx'))]
    hints, hint_source = names_database()
    exports = {}
    for m in bundled:
        for symbol in m['symbols']:
            if symbol['section'] and symbol['binding'] in (1, 2) and symbol['identity']:
                exports.setdefault(identity_key(symbol), []).append(Path(m['path']).name)
    rows = []
    counts = Counter()
    for m in [main] + bundled:
        referenced = Counter(sym for _, kind, sym, _ in m['relocs'] if kind != 8)
        for i, n in referenced.items():
            sym = m['symbols'][i]
            if sym['section'] or not sym['identity']:
                continue
            ident = sym['identity']
            api = hints.get(ident['nid'], '')
            lib = ident['library']['name']
            row = dict(owner=Path(m['path']).name, symbol=sym['name'], nid=ident['nid'],
                       api_hint=api, library=lib, library_version=ident['library']['version_raw'],
                       module=ident['module']['name'], module_version=ident['module']['version_raw'],
                       symbol_type=sym['type'], relocations=n, category=category(lib, api),
                       bundled_exact_candidates=';'.join(exports.get(identity_key(sym), [])),
                       runtime_status='unimplemented_trap')
            rows.append(row)
            if m is main:
                counts[lib] += 1
    out.mkdir(parents=True, exist_ok=True)
    with (out / 'imports.csv').open('w', newline='', encoding='utf-8-sig') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    probe = write_probe(main, out)
    modules = []
    for m in [main] + bundled:
        tls = [p for p in m['ph'] if p['type'] == 7]
        modules.append(dict(path=m['path'], sha256=m['sha256'], elf_sha256=m['elf_sha256'],
                            entry=hex(m['header'][4]), tls=tls, needed=m['needed'],
                            program_headers=m['ph'], missing_metadata=m['missing'],
                            relocation_types=dict(Counter(k for _, k, _, _ in m['relocs'])),
                            libraries=m['libraries'], module_identities=m['modules']))
    report = dict(status='prepared_only', title=meta, reference_commit=REFERENCE_COMMIT,
                  probe=probe, modules=modules, api_name_hint_source=hint_source,
                  main_imports_by_library=dict(counts), import_rows=len(rows),
                  interpretation='Relocation-backed imports; API names are NID hints, not proof of runtime use. '
                  'Exact bundled exports are candidates, not linked/validated implementations. '
                  'Only the main eboot is in boot.bin; all imports trap; TLS is not installed.')
    (out / 'analysis.json').write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(dict(title=meta.get('TITLE'), entry=probe['entry'], image_bytes=probe['size'],
                         main_imports=len(probe['imports']), relocations=probe['relocations'],
                         modules=len(modules), output=str(out)), ensure_ascii=False, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--game', type=Path, default=DEFAULT_GAME)
    parser.add_argument('--out', type=Path, default=ROOT / 'out')
    args = parser.parse_args()
    try:
        prepare(args.game, args.out)
    except (ValueError, OSError, KeyError, IndexError, StopIteration, struct.error) as e:
        parser.exit(1, f'Preparation blocked: {e}\n')
