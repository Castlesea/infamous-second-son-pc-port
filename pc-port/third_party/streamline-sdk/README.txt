NVIDIA Streamline SDK 2.14.16 public headers (MIT, see LICENSE.txt), from
https://github.com/NVIDIA-RTX/Streamline/releases/tag/v2.14.16 (streamline-sdk-v2.14.16.zip), as vendored by
axylpup/shadPS4 (externals/streamline-sdk).
Only these headers are compiled in (ISS_DLSS, pc-port/gpu/iss_dlss_sl.cpp); no NVIDIA binary is linked or
shipped. At run time the player's own signed NVIDIA DLLs are loaded from a dlss folder next to iss-probe.exe
(bin\dlss) or in the game folder (dlss): sl.interposer.dll, sl.common.dll, sl.dlss.dll, nvngx_dlss.dll.
Without them the game's own anti-aliasing runs.
