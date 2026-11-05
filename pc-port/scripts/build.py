# SPDX-License-Identifier: GPL-2.0-or-later
"""Build only the separate Windows probe; never rebuild or install shadPS4."""
import os
from pathlib import Path
import subprocess
import sys
import csv
import json
import shutil
import argparse

import toolchain

ROOT = Path(__file__).resolve().parents[1]
VARS = toolchain.vcvars()
CLANG = toolchain.clang()

def build(output=None):
    if not VARS.exists() or not CLANG.exists():
        raise RuntimeError('Requires Visual Studio with the C++ tools (vcvars64.bat) and clang-cl '
                           '(tools/llvm/bin or the VS "C++ Clang Compiler for Windows" component)')
    env = toolchain.environment(CLANG)
    out = ROOT / 'out'
    out.mkdir(exist_ok=True)
    executable = Path(output).resolve() if output is not None else out/'iss-probe.exe'
    executable.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(ROOT.parent/'build-sdl/SDL3.dll',out/'SDL3.dll')
    from content_profile import create
    create()
    native_objects=[]
    for name in ('reduction','feedback_identity','hooks','geometry','geometry_hook','engine_draw','ultrawide','late_reads'):
        obj=out/f'native-{name}.obj'
        subprocess.run([str(CLANG),'/nologo','/std:c++20','/EHsc','/W4','/WX','/O2','/MD','/D_CRT_SECURE_NO_WARNINGS','/arch:AVX2','/fp:precise','/c',str(ROOT/f'native/{name}.cpp'),f'/Fo:{obj}'],cwd=out,env=env,check=True)
        native_objects.append(str(obj))
    feedback_stub=out/'native-feedback_identity_stub.obj'
    subprocess.run([str(CLANG),'/nologo','/c',str(ROOT/'native/feedback_identity_stub.S'),
                    f'/Fo:{feedback_stub}'],cwd=out,env=env,check=True)
    native_objects.append(str(feedback_stub))
    feedback_test_stub=out/'test-feedback_identity_abi_test.obj'
    subprocess.run([str(CLANG),'/nologo','/c',str(ROOT/'tests/feedback_identity_abi_test.S'),
                    f'/Fo:{feedback_test_stub}'],cwd=out,env=env,check=True)
    subprocess.run([str(CLANG),'/std:c++20','/EHsc','/W4','/WX','/O2','/MD','/arch:AVX2',
                    '/clang:-fuse-ld=lld',str(ROOT/'tests/feedback_identity_abi_test.cpp'),
                    native_objects[1],str(feedback_stub),str(feedback_test_stub),
                    f'/Fe:{out / "feedback-identity-abi-test.exe"}'],cwd=out,env=env,check=True)
    subprocess.run([str(out/'feedback-identity-abi-test.exe')],cwd=out,check=True,timeout=15)
    boot=(out/'boot-linked.bin').read_bytes()
    image_size=json.loads((out/'link.json').read_text(encoding='utf-8'))['image_size']
    start=len(boot)-image_size+0x171610
    (out/'reduction-original.bin').write_bytes(boot[start:start+644])
    subprocess.run([str(CLANG),'/std:c++20','/EHsc','/W4','/WX','/O2','/MD','/arch:AVX2','/fp:precise','/D_CRT_SECURE_NO_WARNINGS','/clang:-fuse-ld=lld',str(ROOT/'tests/native_reduction_test.cpp'),native_objects[0],f'/Fe:{out / "native-reduction-test.exe"}'],cwd=out,env=env,check=True)
    subprocess.run([str(out/'native-reduction-test.exe')],cwd=out,check=True,timeout=15)
    from test_native_geometry import run as test_native_geometry
    test_native_geometry()
    symbols = {}
    with (out / 'linked-imports.csv').open(encoding='utf-8-sig') as f:
        for row in csv.DictReader(f):
            if row['api_hint']:
                symbols[row['symbol']] = row['api_hint']
    (out / 'api_names.h').write_text('// Generated from the local GPL AeroLib-backed import inventory.\n'
        + 'static const struct { const char *identity, *api; } api_names[] = {\n'
        + ''.join('{' + json.dumps(k) + ',' + json.dumps(v) + '},\n' for k, v in sorted(symbols.items()))
        + '};\n', encoding='utf-8')
    cmd = [str(CLANG), '/TC', '/std:c11', '/W4', '/WX', '/O2', '/MD',
           '/clang:-fuse-ld=lld', f'/I{out}', '/D_CRT_SECURE_NO_WARNINGS', '/DZYDIS_STATIC_BUILD', '/DZYCORE_STATIC_BUILD',
           f'/I{ROOT.parent / "research/shadPS4-perf/externals/zydis/include"}',
           f'/I{ROOT.parent / "research/shadPS4-perf/externals/zydis/dependencies/zycore/include"}',
           f'/I{ROOT.parent / "research/shadPS4-perf/externals/sdl3/include"}',
           *[str(ROOT / 'src' / name) for name in ('probe.c','runtime.c','memory.c','sync.c','kernel_sync.c','threads.c','time.c','files.c','services.c','dialogs.c','save_dialog.c','np.c','network.c','http.c','saves.c','trophies.c','pad.c','pad_panel.c','audio.c','gpu.c')],
           str(ROOT.parent/'build-sdl/SDL3.lib'),
           *native_objects,
           str(ROOT.parent / 'build-opt/externals/zydis/Zydis.lib'),
           str(ROOT.parent / 'build-opt/externals/zydis/zycore/Zycore.lib'),
           f'/Fe:{executable}',
           '/link', 'user32.lib', 'gdi32.lib', 'ws2_32.lib', 'xinput.lib', 'winhttp.lib', '/INCREMENTAL:NO', '/DYNAMICBASE', '/NXCOMPAT','/STACK:0x800000,0x200000']
    subprocess.run(cmd, cwd=out, env=env, check=True)
    print(f'Built {executable}')
    test_cmd = cmd[:]
    test_cmd[test_cmd.index(str(ROOT / 'src/probe.c'))] = str(ROOT / 'tests/runtime_test.c')
    test_cmd.insert(1, f'/I{ROOT / "src"}')
    test_cmd[test_cmd.index(f'/Fe:{executable}')] = f'/Fe:{out / "runtime-test.exe"}'
    subprocess.run(test_cmd, cwd=out, env=env, check=True)
    subprocess.run([str(out / 'runtime-test.exe')], cwd=out, check=True, timeout=10)
    sync_cmd=test_cmd[:]
    sync_cmd[sync_cmd.index(str(ROOT / 'tests/runtime_test.c'))]=str(ROOT / 'tests/sync_test.c')
    sync_cmd[sync_cmd.index(f'/Fe:{out / "runtime-test.exe"}')]=f'/Fe:{out / "sync-test.exe"}'
    subprocess.run(sync_cmd,cwd=out,env=env,check=True)
    subprocess.run([str(out / 'sync-test.exe')],cwd=out,check=True,timeout=10)
    pad_cmd=test_cmd[:]
    pad_cmd[pad_cmd.index(str(ROOT/'tests/runtime_test.c'))]=str(ROOT/'tests/pad_test.c')
    pad_cmd[pad_cmd.index(f'/Fe:{out / "runtime-test.exe"}')]=f'/Fe:{out / "pad-test.exe"}'
    subprocess.run(pad_cmd,cwd=out,env=env,check=True)
    subprocess.run([str(out/'pad-test.exe')],cwd=out,check=True,timeout=10)
    save_cmd=test_cmd[:]
    save_cmd[save_cmd.index(str(ROOT/'tests/runtime_test.c'))]=str(ROOT/'tests/save_dialog_test.c')
    save_cmd[save_cmd.index(f'/Fe:{out / "runtime-test.exe"}')]=f'/Fe:{out / "save-dialog-test.exe"}'
    subprocess.run(save_cmd,cwd=out,env=env,check=True)
    subprocess.run([str(out/'save-dialog-test.exe')],cwd=out,check=True,timeout=10)
    search_cmd=test_cmd[:]
    search_cmd[search_cmd.index(str(ROOT/'tests/runtime_test.c'))]=str(ROOT/'tests/save_search_test.c')
    search_cmd[search_cmd.index(f'/Fe:{out / "runtime-test.exe"}')]=f'/Fe:{out / "save-search-test.exe"}'
    subprocess.run(search_cmd,cwd=out,env=env,check=True)
    subprocess.run([str(out/'save-search-test.exe')],cwd=out,check=True,timeout=10)
    subprocess.run([str(CLANG),'/std:c++20','/EHsc','/W4','/WX','/O2','/MD','/clang:-fuse-ld=lld',str(ROOT/'tests/readback_hint_test.cpp'),f'/Fe:{out / "readback-hint-test.exe"}'],cwd=out,env=env,check=True)
    subprocess.run([str(out/'readback-hint-test.exe')],cwd=out,check=True,timeout=10)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, help='Build the EXE separately from out/iss-probe.exe')
    args = parser.parse_args()
    try:
        build(args.output)
    except (OSError, RuntimeError, subprocess.CalledProcessError) as e:
        sys.exit(f'Build failed: {e}')
