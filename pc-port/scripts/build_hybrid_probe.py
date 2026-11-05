"""Build the hook-enabled probe entirely in a fresh experimental directory."""
import argparse
from pathlib import Path
import subprocess
import build_gpu

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    exe = args.output.resolve()
    if not exe.is_relative_to((ROOT / 'out/experimental').resolve()) or exe.exists():
        parser.error('Use a new EXE under out/experimental')
    tasks = subprocess.check_output(['tasklist', '/FI', 'IMAGENAME eq iss-probe.exe', '/FO', 'CSV'], text=True)
    print(tasks, flush=True)
    if 'iss-probe.exe' in tasks.lower():
        raise SystemExit('Jogo em execução; aguardar.')
    exe.parent.mkdir(parents=True, exist_ok=True)
    env = build_gpu.environment()
    clang = str(build_gpu.CLANG)
    objects = []
    for name in ('reduction', 'feedback_identity', 'hooks', 'geometry', 'geometry_hook', 'engine_draw', 'ultrawide', 'late_reads'):
        obj = exe.parent / ('native-' + name + '.obj')
        subprocess.run([clang, '/nologo', '/std:c++20', '/EHsc', '/W4', '/WX', '/O2', '/MD',
                        '/D_CRT_SECURE_NO_WARNINGS', '/arch:AVX2', '/fp:precise', '/c',
                        str(ROOT / 'native' / (name + '.cpp')), '/Fo:' + str(obj)],
                       cwd=exe.parent, env=env, check=True)
        objects.append(str(obj))
    stub = exe.parent / 'native-feedback_identity_stub.obj'
    subprocess.run([clang, '/nologo', '/c', str(ROOT / 'native/feedback_identity_stub.S'),
                    '/Fo:' + str(stub)], cwd=exe.parent, env=env, check=True)
    objects.append(str(stub))
    sources = ('probe.c', 'runtime.c', 'memory.c', 'sync.c', 'kernel_sync.c', 'threads.c',
               'time.c', 'files.c', 'services.c', 'dialogs.c', 'save_dialog.c', 'np.c',
               'network.c', 'http.c', 'saves.c', 'trophies.c', 'pad.c', 'audio.c', 'gpu.c')
    work = ROOT.parent
    command = [clang, '/nologo', '/TC', '/std:c11', '/W4', '/WX', '/O2', '/MD',
               '/clang:-fuse-ld=lld', '/I' + str(ROOT / 'out'), '/D_CRT_SECURE_NO_WARNINGS',
               '/DZYDIS_STATIC_BUILD', '/DZYCORE_STATIC_BUILD',
               '/I' + str(work / 'research/shadPS4-perf/externals/zydis/include'),
               '/I' + str(work / 'research/shadPS4-perf/externals/zydis/dependencies/zycore/include'),
               '/I' + str(work / 'research/shadPS4-perf/externals/sdl3/include'),
               *[str(ROOT / 'src' / name) for name in sources], str(work / 'build-sdl/SDL3.lib'),
               *objects, str(work / 'build-opt/externals/zydis/Zydis.lib'),
               str(work / 'build-opt/externals/zydis/zycore/Zycore.lib'), '/Fe:' + str(exe),
               '/link', 'user32.lib', 'gdi32.lib', 'ws2_32.lib', 'xinput.lib', 'winhttp.lib',
               '/INCREMENTAL:NO', '/DYNAMICBASE', '/NXCOMPAT', '/STACK:0x800000,0x200000']
    subprocess.run(command, cwd=exe.parent, env=env, check=True)
    print('Built', exe)


if __name__ == '__main__':
    main()
