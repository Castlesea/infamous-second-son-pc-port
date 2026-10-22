# SPDX-License-Identifier: GPL-2.0-or-later
"""Build an isolated graphics DLL against the existing Windows renderer dependencies."""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
import os
from pathlib import Path
import re
import shlex
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[1]
WORKSPACE=ROOT.parent
SOURCE=WORKSPACE/'research/shadPS4-perf/src'
OUT=ROOT/'out/gpu-build'
import toolchain
CLANG=toolchain.clang()
LTO_MODES=('none','thin','full')
CPU_TARGETS=('generic','znver3')

def optimization_options(lto='none',cpu='generic',jobs=4):
    if lto not in LTO_MODES or cpu not in CPU_TARGETS:
        raise ValueError('Unsupported LTO mode or CPU target')
    # Keep the working non-LTO object cache and its flag signature unchanged.
    directory=OUT if (lto,cpu)==('none','generic') else OUT/'variants'/f'lto-{lto}_cpu-{cpu}'
    compile_flags=[]
    link_flags=[]
    if lto!='none':
        compile_flags.append(f'/clang:-flto={lto}')
        # clang-cl /O2 selects LLVM -O3. Match it in the LTO backend without fast math.
        link_flags += ['/opt:lldlto=3',f'/opt:lldltojobs={jobs}']
        if lto=='thin':link_flags.append('/lldltocache:'+str(directory/'thinlto-cache'))
    if cpu!='generic':compile_flags.append(f'/clang:-march={cpu}')
    if os.environ.get('ISS_GPU_LINE_TABLES')=='1':
        # Profiling only: the same code with line tables in a PDB, in its own object cache.
        directory=directory.with_name(directory.name+'-lines')
        compile_flags+=['/Z7','/clang:-gline-tables-only']
        link_flags+=['/DEBUG:FULL']
    return directory,compile_flags,link_flags

def link_command(objects,libraries,dll_output,link_flags):
    command=[str(CLANG),'/LD','/clang:-fuse-ld=lld',*[str(p) for p in objects],*libraries,
             f'/Fe:{dll_output}','/link','/INCREMENTAL:NO','/DYNAMICBASE','/NXCOMPAT',
             '/EXPORT:iss_gpu_initialize','/EXPORT:iss_gpu_resolve','/EXPORT:iss_gpu_fault','/EXPORT:iss_gpu_map','/EXPORT:iss_gpu_readback','/EXPORT:iss_gpu_invalidate',
             '/EXPORT:iss_gpu_feedback_identity','/EXPORT:iss_gpu_file_opened',*link_flags]
    command.append('/MAP:'+str(dll_output.with_suffix('.map')))
    return command

def environment():
    return toolchain.environment(CLANG)

def build(check=False,jobs=4,output=None,lto='none',cpu='generic',dry_run=False):
    out,optimization_flags,link_flags=optimization_options(lto,cpu,jobs)
    ninja=(WORKSPACE/'build-opt/build.ninja').read_text(encoding='utf-8')
    defs=shlex.split(re.search(r'^  DEFINES = (.+)$',ninja,re.M).group(1))
    # Paths with forward slashes: some CMake versions write backslashes, which shlex.split drops.
    includes=shlex.split(re.search(r'^  INCLUDES = (.+)$',ninja,re.M).group(1).replace('\\','/'))
    common=['decoder','error','io_file','native_clock','number_utils','rdtsc','spin_lock','recursive_lock','string_util','thread','signal_context','ntapi']
    sources=list((SOURCE/'video_core').rglob('*.cpp'))+list((SOURCE/'shader_recompiler').rglob('*.cpp'))
    sources=[p for p in sources if p.name!='renderdoc.cpp']
    overrides={'video_core/buffer_cache/buffer_cache.cpp':'buffer_cache.cpp','video_core/renderer_vulkan/vk_rasterizer.cpp':'vk_rasterizer.cpp','video_core/renderer_vulkan/vk_pipeline_cache.cpp':'vk_pipeline_cache.cpp','video_core/renderer_vulkan/vk_graphics_pipeline.cpp':'vk_graphics_pipeline.cpp','video_core/renderer_vulkan/vk_pipeline_serialization.cpp':'vk_pipeline_serialization.cpp','video_core/renderer_vulkan/vk_runtime.cpp':'vk_runtime.cpp','video_core/amdgpu/liverpool.cpp':'liverpool.cpp','video_core/page_manager.cpp':'page_manager.cpp','video_core/renderer_vulkan/vk_instance.cpp':'vk_instance.cpp','video_core/renderer_vulkan/vk_resource_pool.cpp':'vk_resource_pool.cpp','video_core/buffer_cache/buffer.cpp':'buffer.cpp','video_core/renderer_vulkan/vk_presenter.cpp':'vk_presenter.cpp','video_core/renderer_vulkan/vk_scheduler.cpp':'vk_scheduler.cpp','video_core/renderer_vulkan/vk_staging_buffer_pool.cpp':'vk_staging_buffer_pool.cpp','video_core/texture_cache/texture_cache.cpp':'texture_cache.cpp','shader_recompiler/ir/passes/flatten_extended_userdata_pass.cpp':'flatten_extended_userdata_pass.cpp','shader_recompiler/recompiler.cpp':'recompiler.cpp','shader_recompiler/ir/passes/resource_patching_pass.cpp':'resource_patching_pass.cpp','shader_recompiler/frontend/fetch_shader.cpp':'fetch_shader.cpp','video_core/cache_storage.cpp':'cache_storage.cpp','video_core/renderer_vulkan/vk_swapchain.cpp':'vk_swapchain.cpp','video_core/texture_cache/image.cpp':'image.cpp'}
    sources=[ROOT/'gpu/overrides'/overrides[p.relative_to(SOURCE).as_posix()] if p.relative_to(SOURCE).as_posix() in overrides else p for p in sources]
    sources += [SOURCE/f'common/{n}.cpp' for n in common]
    sources += [SOURCE/'common/logging/classes.cpp',ROOT/'gpu/overrides/gnmdriver.cpp',
                ROOT/'gpu/overrides/videoout_driver.cpp',SOURCE/'core/libraries/videoout/video_out.cpp',
                SOURCE/'core/libraries/kernel/equeue.cpp']
    sources += [SOURCE/'imgui/renderer/imgui_impl_vulkan.cpp',SOURCE/'imgui/renderer/texture_manager.cpp',SOURCE/'common/stb.cpp']
    sources += list((SOURCE/'core/libraries/ajm').glob('*.cpp'))
    sources += [SOURCE/'core/libraries/videodec'/n for n in ('videodec.cpp','videodec_impl.cpp','video_utils.cpp')]
    sources += list((ROOT/'gpu').glob('*.cpp'))
    if check:
        sources=[ROOT/'gpu/overrides/gnmdriver.cpp',ROOT/'gpu/overrides/vk_presenter.cpp',
                 ROOT/'gpu/overrides/liverpool.cpp',ROOT/'gpu/overrides/vk_scheduler.cpp',
                 ROOT/'gpu/overrides/texture_cache.cpp']+list((ROOT/'gpu').glob('*.cpp'))
    flags=['/nologo','/TP','/std:c++latest','/EHsc','/MD','/O2','/DNDEBUG','/clang:-Wno-deprecated-declarations',
           f'/I{ROOT / "gpu/shim"}',f'/I{ROOT / "gpu"}',
           f'/I{SOURCE / "core/libraries/gnmdriver"}']+defs+includes+optimization_flags
    libs=shlex.split(re.search(r'^  LINK_LIBRARIES = (.+)$',ninja,re.M).group(1).replace('\\','/'))
    # Dependencies come from the known working build; no emulator objects/main are linked.
    libraries=[]
    for lib in libs:
        path=Path(lib)
        if path.is_absolute():libraries.append(str(path))
        elif (WORKSPACE/'build-opt'/path).exists():libraries.append(str(WORKSPACE/'build-opt'/path))
        else:libraries.append(lib)
    dll_output=Path(output).resolve() if output is not None else ROOT/'out/iss-gpu.dll'
    configuration={'lto':lto,'cpu':cpu,'object_directory':str(out),
                   'compile_flags':flags,'link_flags':link_flags}
    if dry_run:
        planned_objects=[out/(hashlib.sha256(str(source).encode()).hexdigest()[:16]+'.obj') for source in sources]
        print(json.dumps({**configuration,'source_count':len(sources),
            'sources':[str(source) for source in sources],
            'example_compile_command':[str(CLANG),*flags,'/c',str(sources[0]),f'/Fo:{planned_objects[0]}'],
            'link_command':None if check else link_command(planned_objects,libraries,dll_output,link_flags)},indent=2))
        return 0
    out.mkdir(parents=True,exist_ok=True)
    (out/'build-config.json').write_text(json.dumps(configuration,indent=2),encoding='utf-8')
    env=environment()
    # Header changes invalidate all objects. Source/dependency locations remain read-only.
    signature=hashlib.sha256(('\0'.join(flags)+'\0'+''.join(p.read_text(encoding='utf-8')
        for p in sorted((ROOT/'gpu').rglob('*.h')))).encode()).hexdigest()
    def compile_one(source):
        name=hashlib.sha256(str(source).encode()).hexdigest()[:16]
        obj=out/(name+'.obj'); log=out/(name+'.log'); stamp=out/(name+'.json')
        source_bytes=source.read_bytes()
        if source==ROOT/'gpu/native_shadow.cpp':
            source_bytes+=(ROOT/'gpu/native_shadow_capture.inl').read_bytes()
            source_bytes+=(ROOT/'gpu/native_shadow_outer.inl').read_bytes()
        if source==ROOT/'gpu/overrides/liverpool.cpp':
            source_bytes+=(ROOT/'gpu/hybrid_family_profile.inl').read_bytes()
        # This local implementation fragment is private to the rasterizer, not a global header.
        if source==ROOT/'gpu/overrides/vk_rasterizer.cpp':
            for name in ('draw_pipe_rasterizer.inl', 'native_draw_run_rasterizer.inl',
                         'native_instanced_run_rasterizer.inl', 'native_cpu_draw_run.inl',
                         'stage_bind_memo.inl', 'fast_draw.inl', 'native_items.inl', 'native_visual_probe.inl',
                         'native_depth_run.inl', 'native_depth_run_verify.inl',
                         'native_shadow_rasterizer.inl', 'native_gbuffer_rasterizer.inl',
                         'occlusion_diag.inl','native_material_rasterizer.inl','native_recorder.inl',
                         'native_passes_rasterizer.inl', 'native_compute_rasterizer.inl'):
                fragment=ROOT/'gpu'/name
                if fragment.exists():source_bytes+=fragment.read_bytes()
        identity={'flags':signature,'source_sha256':hashlib.sha256(source_bytes).hexdigest()}
        if obj.exists() and stamp.exists() and json.loads(stamp.read_text())==identity:return source,obj,0,''
        result=subprocess.run([str(CLANG),*flags,'/c',str(source),f'/Fo:{obj}'],env=env,cwd=WORKSPACE/'build-opt',capture_output=True,text=True)
        output=result.stdout+result.stderr;log.write_text(output,encoding='utf-8')
        if result.returncode==0:stamp.write_text(json.dumps(identity),encoding='utf-8')
        return source,obj,result.returncode,output
    objects=[];failed=[]
    with ThreadPoolExecutor(max_workers=jobs) as executor:
        futures=[executor.submit(compile_one,p) for p in sources]
        for future in as_completed(futures):
            source,obj,code,compile_output=future.result()
            if code:
                failed.append(str(source));print(f'FAILED {source.name}\n{compile_output[:6500]}',flush=True)
            else:objects.append(obj);print(f'Compiled {source.name}',flush=True)
    if failed:
        (out/'failed.json').write_text(json.dumps(failed,indent=2),encoding='utf-8');return 1
    if check:return 0
    dll_output.parent.mkdir(parents=True,exist_ok=True)
    command=link_command(objects,libraries,dll_output,link_flags)
    response=out/'link.rsp'
    response.write_text(subprocess.list2cmdline(command[1:]),encoding='utf-8')
    result=subprocess.run([str(CLANG),'@'+str(response)],env=env,cwd=WORKSPACE/'build-opt',capture_output=True,text=True)
    (out/'link.log').write_text(result.stdout+result.stderr,encoding='utf-8')
    if result.returncode == 0:
        dll_output.with_suffix('.symbols.json').write_text(json.dumps({
            'dll_sha256':hashlib.sha256(dll_output.read_bytes()).hexdigest(),
            'map_sha256':hashlib.sha256(dll_output.with_suffix('.map').read_bytes()).hexdigest(),
            'dll_path':str(dll_output),'map_path':str(dll_output.with_suffix('.map')),
            'build_configuration':configuration},indent=2),encoding='utf-8')
    print((result.stdout+result.stderr)[:15000]);return result.returncode

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--check',action='store_true');parser.add_argument('--jobs',type=int,default=4)
    parser.add_argument('--output',type=Path,help='Link a DLL at this path, leaving out/iss-gpu.dll untouched')
    parser.add_argument('--lto',choices=LTO_MODES,default='none',help='LLVM link-time optimization (default: none)')
    parser.add_argument('--cpu',choices=CPU_TARGETS,default='generic',help='CPU target; znver3 specializes for Ryzen Zen 3 (default: generic)')
    parser.add_argument('--dry-run',action='store_true',help='Print the configuration and commands as JSON without compiling or writing build files')
    args=parser.parse_args();raise SystemExit(build(args.check,max(1,min(args.jobs,8)),args.output,args.lto,args.cpu,args.dry_run))
