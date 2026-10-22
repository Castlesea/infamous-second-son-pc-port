# SPDX-License-Identifier: GPL-2.0-or-later
"""Builds the portable release folder (the user zips it):
python scripts/build_release.py [--name NAME]

<out>/release/<NAME>/
  inFAMOUS Second Son Launcher.exe
  bin/            iss-probe.exe, iss-gpu.dll, SDL3.dll
  assets/fonts/   Oswald.ttf, OFL.txt
  shaders/recipes/ the shader recipes: which pipelines the game uses (hashes, render states,
                  vertex formats), no shader code; the port rebuilds them from the user's files
  README.txt, LICENSE.txt, THIRD_PARTY.txt
No game file, no shader derived from the game, no user data: the launcher asks for the user's own
game folder on first launch and prepares it (launcher/prepare_game.cpp)."""
import argparse
import hashlib
from pathlib import Path
import re
import shutil
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
EXP = ROOT / 'out/experimental/20261016-native-passes-v1'
# porte-vsync1 (2026-10-19): porte-x-save + shaders ahead of time + the 75 Hz stutter fixes;
# porte-vsync1-pad: the same renderer, the controller layout, touchpad window, mouse touchpad and
# spray can (src/pad.c, src/pad_panel.c).
GPU_DLL = EXP / 'base-20261019-porte-vsync1-pad/iss-gpu.dll'
PROBE_EXE = EXP / 'base-20261019-porte-vsync1-pad/iss-probe.exe'
RECIPES = EXP / 'base-20261019-porte-aot3/recipes'          # verified stages only (make_recipes.py)
SDL = ROOT / 'out/SDL3.dll'

README = """inFAMOUS Second Son - PC port (unofficial)
===========================================

A native PC port of inFAMOUS Second Son's renderer (Vulkan). It does NOT include the game:
you need your own copy.

Requirements
- Windows 10/11 64-bit, or a Steam Deck / Linux with Proton.
- A GPU with Vulkan 1.3.
- Your own decrypted dump of inFAMOUS Second Son (CUSA00004 Europe or CUSA00223 North America,
  the same eboot.bin), the retail build this port was made for: the folder that contains eboot.bin, sce_sys and sce_module. The launcher checks it.

How to play
1. Extract this folder anywhere you can write to (not inside Program Files).
2. Run "inFAMOUS Second Son Launcher.exe".
3. Pick your game folder. The launcher prepares it once (a few seconds); your game folder is
   only read, never modified.
4. PLAY.

The first launch prepares the game's shaders for your GPU ("Preparing shaders (first run
only)" on the loading screen, about 2 minutes); later launches take a few seconds. A few areas
may still stutter the first time you see them; it gets smoother every session (the cache is kept
in user\\shader-cache).

DLC
Settings > Advanced > DLC folder: pick the folder with your inFAMOUS Second Son DLC .pkg files.
They are read in place (the costumes and Cole's Legacy are unlocked); nothing is installed into
the game folder.

Controls
- Launcher: mouse, keyboard (arrows, Enter, Esc, Tab) or controller (A, B, LB/RB, Start).
- In game: a controller (Xbox pads, PlayStation pads, Steam Input / Steam Deck) or the keyboard.
  F9 shows the frame rate, Alt+Enter toggles fullscreen.
- Controllers without a touchpad: the game asks for the touchpad in fast moments of gameplay.
  By default View and Menu press its left and right side; CONTROLS in the launcher changes what
  each button does. PlayStation controllers use their own touchpad.
- F5: the touchpad window (a small touchpad to press and drag with the mouse, and buttons for
  the sides, click, swipes, shake and recenter). It stays on top without taking the focus.
- F6: the mouse is the touchpad (left button: a finger at the cursor, right button: click).
- F7: spray can, for the graffiti missions: the mouse moves the can, left button sprays, right
  button shakes it, middle button recenters, Shift aims finely.

Keyboard (while the game window has the focus)
- Move: W A S D. Camera: I J K L.
- Cross / Circle / Square / Triangle: N / B / V / C (or numpad 2 / 6 / 4 / 8).
- L1 / R1: Q / U.  L2 / R2: E / O.  L3 / R3: X / M.
- D-pad: arrow keys.  Options (pause): Esc.
- Touchpad: Space or Tab (left side), Backspace (right side), F8 (both), T (middle);
  1 / 2 / 3 / 4 swipe up / down / left / right.
- F5 touchpad & spray window, F6 mouse as touchpad, F7 spray mode, F9 frame rate,
  Alt+Enter fullscreen.
Frame rate
Settings > Display > Frame rate limit (Unlimited, 30 to 240). Above your monitor's refresh rate
it needs VSync Off.

Steam Deck
Add "inFAMOUS Second Son Launcher.exe" to Steam as a non-Steam game and run it with Proton. The
launcher opens fullscreen and is fully usable with the Deck's controls.

Files
- user\\launcher.ini  settings
- user\\saves         your saves
- user\\logs          logs of the last sessions (attach them to bug reports)

Known issues
- Some objects and shadows switch level of detail close to the camera (pop-in).
- The frame rate can dip in the densest city areas.

This project is not affiliated with Sony Interactive Entertainment or Sucker Punch Productions.
inFAMOUS is a trademark of Sony Interactive Entertainment. No game files are distributed.
"""

THIRD_PARTY = """Third-party software in this release
=====================================

- shadPS4 (https://github.com/shadps4-emu/shadPS4): the Vulkan renderer, shader recompiler and
  GPU command processing in bin\\iss-gpu.dll are built from shadPS4 sources, GPL-2.0-or-later.
- bloodborne_pc (https://github.com/deadinside28/bloodborne_pc): the PS4 SELF reading and module
  linking of the launcher's game preparation are adapted from its scripts, GPL-2.0-or-later.
- SDL 3 (https://libsdl.org) in bin\\SDL3.dll, zlib license.
- Oswald font (https://github.com/googlefonts/OswaldFont), SIL Open Font License 1.1
  (assets\\fonts\\OFL.txt).

This port is distributed under GPL-2.0-or-later (LICENSE.txt). Its source code is available from
the project's repository.
"""


# The absolute source paths of the machine a binary was built on, which compilers leave in assert and
# log messages ("... at C:/<folders>/research/shadPS4-perf/externals/imgui/imgui.cpp:20899"): cut
# down to the path inside the project, so a release names no local folder or user. Only the text of
# those NUL-terminated messages changes (the rest of each string moves left, NUL-padded).
PROJECT_DIRS = (b'research', b'pc-port', b'build-opt', b'build-sdl', b'tools', b'launcher')
ABS_PATH = re.compile(rb'[A-Za-z]:[\\/][\x20-\x7e]*?[\\/](?=(?:' + b'|'.join(PROJECT_DIRS) + rb')[\\/])')
LOCAL_PATH = re.compile(rb'(?i)[A-Za-z]:[\\/](?:Users|Documents and Settings)[\\/]')


def scrub_paths(path):
    data = bytearray(path.read_bytes())
    count = 0
    for m in reversed(list(ABS_PATH.finditer(data))):
        end = data.find(b'\0', m.end())
        if end < 0:
            continue
        data[m.start():end] = bytes(data[m.end():end]) + b'\0' * (m.end() - m.start())
        count += 1
    if count:
        path.write_bytes(bytes(data))
    left = len(LOCAL_PATH.findall(bytes(data)))
    print(f'{path.name}: {count} build paths cut' + (f', {left} user paths LEFT' if left else ''))
    return left


def build(name):
    for f in (GPU_DLL, PROBE_EXE, SDL, RECIPES / 'manifest.json'):
        if not f.is_file():
            sys.exit(f'missing {f}')
    out = ROOT / 'out' / 'release' / name
    if out.exists():
        shutil.rmtree(out)
    (out / 'bin').mkdir(parents=True)
    subprocess.run([sys.executable, str(ROOT / 'scripts/build_launcher.py'), '--output',
                    str(out / 'inFAMOUS Second Son Launcher.exe')], check=True)
    shutil.copy2(PROBE_EXE, out / 'bin/iss-probe.exe')
    shutil.copy2(GPU_DLL, out / 'bin/iss-gpu.dll')
    shutil.copy2(SDL, out / 'bin/SDL3.dll')
    shutil.copytree(ROOT / 'launcher/assets/fonts', out / 'assets/fonts')
    shutil.copytree(RECIPES, out / 'shaders/recipes')
    (out / 'README.txt').write_text(README, encoding='utf-8')
    (out / 'THIRD_PARTY.txt').write_text(THIRD_PARTY, encoding='utf-8')
    shutil.copy2(ROOT / 'LICENSE', out / 'LICENSE.txt')
    if sum(scrub_paths(f) for f in sorted(out.rglob('*')) if f.suffix.lower() in ('.exe', '.dll')):
        sys.exit('a binary still names a user folder')
    manifest = []
    for f in sorted(out.rglob('*')):
        if f.is_file():
            manifest.append(f'{hashlib.sha256(f.read_bytes()).hexdigest()[:16]}  {f.stat().st_size:>10}  {f.relative_to(out)}')
    print('\n'.join(manifest))
    print(f'Release folder: {out}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--name', default='inFAMOUS-Second-Son-PC-Port-' + time.strftime('%Y%m%d'))
    build(parser.parse_args().name)
