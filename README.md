# inFAMOUS Second Son — native PC port (Windows)

![Delsin at a D.U.P. checkpoint in the forest, 123 FPS](docs/media/forest-checkpoint.jpg)
*Ryzen 7 7800X3D + Radeon RX 9070 XT, Vulkan, VSync off: 123 FPS (7.5 ms a frame).*

| | |
|---|---|
| ![Seattle street in the rain, 91 FPS](docs/media/city-street.jpg) | ![Seattle rooftops at 74 FPS on a 75 Hz display](docs/media/city-rooftop.jpg) |
| *Seattle in the rain, VSync off: 91 FPS.* | *Rooftops on a 75 Hz display, VSync on: 74 FPS.* |

An experimental native Windows port of the PS4 game *inFAMOUS Second Son*.

> **You need your own copy of the game.** No game files, keys or copyrighted assets are part of this repository or its releases.

## Video showcase

### [▶ Watch the video showcase on YouTube](https://youtu.be/V8LJlxHLb_Q)

## The idea

Most ways of playing a console game on PC emulate the console: they translate its CPU code and re-interpret, command by command, everything it asks of the GPU. This project goes the other way, and moves the game onto the PC natively, piece by piece:

- **CPU: already native.** The game's own x86-64 code runs directly on the host processor. A custom runtime replaces the PS4 system libraries (memory, threads, files, saves, audio, the controller), so no CPU emulation is involved.
- **GPU: becoming native.** The renderer started from [shadPS4](https://github.com/shadps4-emu/shadPS4)'s video core, which reads the PS4 GPU command stream and translates it to Vulkan. Step by step, the engine's rendering is taken off that translation: its draws and dispatches are recorded straight into Vulkan command buffers when the game submits them ("native passes"), and its shaders are compiled ahead of time instead of while playing.
- **The goal:** stop depending on emulation and translation altogether, so the game answers like a PC game. Less work per frame means higher frame rates, lower input latency and fewer stutters, and room for PC options (frame rate limits, unlocked frame rates, mouse and keyboard).

What is still translated today is the rest of the GPU command processor; moving it to native is the next step (see [Roadmap](#roadmap)).

## Status

- Playable: story saves, free roam, menus, pause and saving work.
- Reference machine: Ryzen 7 7800X3D + Radeon RX 9070 XT. With VSync off:
  - **Smaller scenes run above 100 FPS**: 123 FPS at the forest checkpoint.
  - **The open city runs at about 80–95 FPS**: 91 FPS in the street above.
  - **The densest city views drop to about 60–70 FPS**: there the command processor (still translated) is the limit, and it is the next thing to move to native.
- On a 75 Hz display with VSync on, the city holds 75 FPS: frames over 25 ms were cut from ~17 to ~1.4 a minute (10-minute scripted run).
- Shaders ahead of time: the game's compute shaders are compiled from the player's own game files before the first session ("Preparing shaders (first run only)", about 2 minutes). Graphics pipelines come from shader recipes (pipeline states and hashes, no shader code), rebuilt from the same files.
- Known open issues:
  - Late appearance or pop-in of some buildings and map structures.
  - Shader compiles in areas no cache has seen yet (a new player's first session).
  - Lower frame rate in the densest views (command processor bound).

## Controls

- **Controllers:** Xbox pads, PlayStation pads and Steam Input / Steam Deck. The keyboard works too.
- **Other keys:** F9 shows the frame rate, Alt+Enter toggles fullscreen.
- **The ? at the top right of the launcher** lists the in-game keys below.

### Controllers without a touchpad

The game asks for the PS4 touchpad in fast moments of gameplay (absorbing powers, finishers, prompts). On a controller without one, **View and Menu press the left and right side of the touchpad** by default, so it stays under your thumbs. **CONTROLS** in the launcher changes what each button does; PlayStation controllers use their own touchpad and keep their layout.

![The launcher's CONTROLS page](docs/media/launcher-controls.png)

### F5: the Touchpad & Spray window

A small window that stays on top of the game **without taking the focus**: the game keeps the keyboard and the controller while you click it. F5 shows or hides it, and each tab has its own guide.

| TOUCHPAD tab | SPRAY tab | Spray mode on |
|---|---|---|
| ![TOUCHPAD tab](docs/media/ui-touchpad.png) | ![SPRAY tab](docs/media/ui-spray.png) | ![Spray mode on](docs/media/ui-spray-on.png) |

**TOUCHPAD tab:**
- **Touchpad:** press and drag on it with the mouse, and the game sees a finger there. Right button: a touchpad click. Shift: two fingers, mirrored.
- **Left / Both / Right / Click:** press that side of the touchpad (Click: the middle), for the game's "press the touchpad" prompts.
- **Swipes:** a finger swipe up, down, left or right, for the swipe prompts.
- **Mouse as touchpad (F6):** in the game window, the left button is a finger at the cursor and the right button clicks.

**SPRAY tab: the spray can of the graffiti missions (F7).** On a PS4 you hold the controller like a spray can and move it. Here:
1. At a graffiti spot, press **START SPRAY MODE** (or F7). The virtual controller is held on its side like a can, and the cursor is hidden in the game window.
2. **Shake the can:** hold the right mouse button.
3. **Spray:** hold the left mouse button and move the mouse to paint. The mouse turns the can using its raw movement, without Windows' pointer acceleration, so the same hand movement always paints the same stroke. Shift aims finely, the middle button recenters, and **Speed** changes how far the can turns.
4. Press **F7** to end spray mode and get the cursor back. While spray mode is on, the window lets the clicks through to the game.

### Keyboard

The whole controller is on the keyboard too, while the game window has the focus:

| Controller | Keys |
|---|---|
| Left stick (move) | W A S D |
| Right stick (camera) | I J K L (I up, K down, J left, L right) |
| Cross / Circle / Square / Triangle | N / B / V / C (or numpad 2 / 6 / 4 / 8) |
| L1 / R1 | Q / U |
| L2 / R2 | E / O |
| L3 / R3 (stick press) | X / M |
| D-pad | Arrow keys |
| Options (pause) | Esc |
| Touchpad: left side / right side / both / middle | Space or Tab / Backspace / F8 / T |
| Touchpad swipes: up / down / left / right | 1 / 2 / 3 / 4 |

| Hotkey | What it does |
|---|---|
| F5 | Shows or hides the Touchpad & Spray window |
| F6 | Mouse as touchpad on / off |
| F7 | Spray mode on / off |
| F9 | Frame rate overlay |
| Alt+Enter | Fullscreen / windowed |

## Launcher

![The launcher's main page with the in-game key guide](docs/media/launcher-help.png)

- Pick your game folder once; the launcher prepares it (your game folder is only read, never modified). With a game folder set, the launcher shows the game's own art from your files.
- DLC: your own DLC packages are read in place (costumes, Cole's Legacy).
- **Display:**
  - window mode and VSync;
  - frame rate limit: Unlimited, or 30 to 240;
  - FSR 1 upscaling.
- **Graphics:**
  - resolution: 16:9 up to 4K, 21:9 (2560x1080, 3440x1440), 24:10 (3840x1600), 32:9 (3840x1080, 5120x1440), or Desktop (your monitor's);
  - field of view: 90% to 130% of the game's;
  - HUD width: on screens wider than 16:9 the HUD edges stay in a centred 16:9 or 21:9 area, or go to the full width.
  - GPU memory: Auto reads the game's memory from copies in video memory on NVIDIA (`ISS_UNIFIED_MEMORY=0`) and in place elsewhere. On an RTX 5070 Ti the in-place reads went over PCIe: the tiled lighting shader took 33-50 ms per frame, 22 FPS instead of 117 at 1080p.

  Wider than 16:9 keeps the game's vertical view and shows more to the sides (Hor+); the menus and HUD keep their size. See [Ultrawide](#ultrawide).
- **Game:** the language, from the 14 the CUSA00004 package ships: English, French, Italian, German, Spanish, Dutch, Portuguese, Russian, Polish and Turkish (text and voices), Danish, Finnish, Norwegian and Swedish (subtitles, English voices).
- **Volume** (GAME tab): 50% to 200% of the game's own level, in steps of 5 (`ISS_VOLUME`). Above 100% the peaks go through a soft limiter instead of clipping.
- **CONTROLS:** the controller layout described above.
- **Shader cache:** it builds up from session to session. A session that crashes is rolled back, so a bad cache never sticks.
## Ultrawide

`pc-port/native/ultrawide.cpp` patches the game before it starts (each patched function is checked against the 1.07 eboot first):
- `ISS_RESOLUTION=WxH`: the renderer's output and render size, which the game otherwise sets to 1920x1080. Every render target is derived from it.
- The cameras take their aspect from the UI screen size: it becomes (1080 x W/H) x 1080, so the vertical field of view stays and the view widens. The UI is laid out from the centre in units of the screen height, so it keeps its size.
- `ISS_HUD_ASPECT=a` (0: full width): the left and right HUD anchors follow the safe area margins, which are moved in to a centred a:1 area. Subtitles wrap inside it too.
- `ISS_FOV=p`: the cameras' tan(fov/2) times p/100.
- Render size limit: the forward shaders walk per-tile light lists (16x16 pixel tiles) until a terminator, and the lists live in a buffer of fixed size. With too many tiles the last lists are never written and the GPU hangs (on NVIDIA and AMD alike): the title screen at 5120x1440, a cutscene at 4984x1400. The picture is rendered with at most as many tiles as the game uses on PS4 Pro (3200x1800, 22600 tiles; `ISS_RENDER_MAX_TILES`), in the same aspect, and the window scales it up: 5120x1440 renders at 4520x1264, 3440x1440 and smaller at their own size.
- Above 1080p the game gets 5.5 GiB of direct memory instead of 4.5 (`ISS_DIRECT_MEMORY_MB`): its main heap, where the render targets live too, ran out at 5120x1440.
- Cutscenes are rendered by the engine at the full width, without black bars. Full-screen UI backdrops made for 16:9 (the title screen's black background, the opening text) cover only the centre.

The launcher sets all three from **SETTINGS > GRAPHICS**. The window keeps the picture's aspect; in fullscreen on a monitor of the same aspect the picture fills it.

## Repository layout

| Path | What it holds |
|---|---|
| `pc-port/src/` | Loader and PS4 runtime (C): memory, threads, files, controller and touchpad window (SDL3, Win32), audio, saves, video glue |
| `pc-port/gpu/` | Renderer: shadPS4 video core overrides (`overrides/`), shims (`shim/`), native renderer (`native_*`) |
| `pc-port/native/` | Native C++ reimplementations of selected engine routines |
| `pc-port/scripts/` | Game preparation, build, play launcher and analysis scripts |
| `pc-port/tests/` | Runtime, controller and renderer unit tests |
| `pc-port/launcher/` | The release launcher (C++, `scripts/build_release.py` packages it) |
| `pc-port/third_party/` | `shadPS4-perf-local.patch`: the changes this port makes to its shadPS4 base |
| `launcher/` | Older PowerShell launcher (development) |

Local-only folders (git-ignored):
- `research/`: the shadPS4 sources the renderer is built from.
- `build-opt/`, `build-sdl/`: CMake builds that supply flags, SDL3 and zydis.
- `tools/`: the LLVM toolchain.
- `pc-port/out/`: builds, runs, caches and saves.

## Building

Requirements:
- Windows 11.
- Visual Studio 2022 or 2026 with "Desktop development with C++" (MSVC, Windows SDK); found with vswhere, or `ISS_VCVARS`.
- LLVM `clang-cl` in `tools/llvm/`, or Visual Studio's "C++ Clang Compiler for Windows" component, or `ISS_CLANG`.
- Python 3.12.
- shadPS4 sources in `research/shadPS4-perf`: [weshayoub92/shadPS4-perf](https://github.com/weshayoub92/shadPS4-perf) at commit `584374867d6a`, with `pc-port/third_party/shadPS4-perf-local.patch` applied (`git apply`). Configure it with CMake into `build-opt/` (renderer flags and zydis) and `build-sdl/` (SDL3).

```bat
cd pc-port\scripts
python prepare.py --game "<your game folder>"         & rem reads eboot/modules/param.sfo only
python link_modules.py --game "<your game folder>"
python build.py                                       & rem iss-probe.exe (loader + runtime) and its tests
python build_gpu.py --jobs 12                         & rem iss-gpu.dll (renderer)
python build_release.py --name <folder>               & rem the release folder (launcher included)
```

## Running

- Release: extract the zip, run `inFAMOUS Second Son Launcher.exe` and pick your game folder.
- Development: `python pc-port/scripts/play_porte.py` starts the current development build.
- The first start prepares the shaders. Later starts load the cache in about a second.

Most behaviour is controlled with `ISS_*` environment variables, set by the launcher and the scripts. `pc-port/scripts/play_porte.py` lists the ones the current build turns on, with the reason and the measurement behind each.

## Roadmap

1. **Finish moving the GPU command processor to native.** This is the last big translated part. It means cheaper native pass execution, no waiting on the recording pool, asynchronous readbacks and the remaining non-native draws. It is the limit in the densest views today.
2. **Small fixes in the same pass:**
   - shader recipe coverage, for fewer compiles in a new player's first session;
   - the remaining streaming-burst frames.
3. **A full code review** and clean-up pass.

## Contributors

- **[Castlesea](https://github.com/Castlesea)**: author and maintainer.
- **Claude** (Anthropic): AI pair programmer. Engineering, profiling and code across the runtime, the renderer and the launcher.

## License and credits

GPL-2.0-or-later (see `pc-port/LICENSE` and `pc-port/THIRD_PARTY.md`). The project builds on:
- the shadPS4 video core and shader recompiler;
- the native-port approach of bloodborne_pc;
- SDL3, Vulkan, Xbyak, zydis and the other libraries listed in `THIRD_PARTY.md`.

*inFAMOUS* is a trademark of Sony Interactive Entertainment. This project is not affiliated with Sony or Sucker Punch Productions.
