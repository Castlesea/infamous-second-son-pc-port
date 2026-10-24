// SPDX-License-Identifier: GPL-2.0-or-later
#include "iss_gpu.h"
#include "native_shadow.h"
#include "native_gbuffer.h"
#include "native_passes.h"
#include "pause_guard.h"
#include "repeat_census.h"
namespace iss::native_gbuffer { int InstallCensus(std::uint64_t base); }
#include "sparse_copy.h"
#include "readback_hint.h"
#include "deep_profile.h"
#include "compute_phase.h"
#include "infamous_renderer/feedback_identity_registry.h"
#include "runtime_timeline.h"
#include <SDL3/SDL.h>
#include <atomic>
// The shader preload's progress (vk_pipeline_serialization.cpp): entries read, phase
// (0 not started, 1 reading, 2 finishing the compiles, 3 done), shown in the window title.
namespace iss { extern std::atomic<unsigned> g_preload_entries, g_preload_total; extern std::atomic<int> g_preload_phase; }
#include <atomic>
#include <algorithm>
#include <cstdlib>
#include <chrono>
#include <condition_variable>
#include <cstring>
#include <filesystem>
#include "g4_conditional_backing_write.h"
#include <functional>
#include <limits>
#include <map>
#include <mutex>
#include <thread>
#include <vector>
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#include <tlhelp32.h>
#include <boost/asio/io_context.hpp>
#include <boost/asio/executor_work_guard.hpp>
#include <xxhash.h>
#include "common/elf_info.h"
#include "common/path_util.h"
#include "common/perf_stats.h"
#include "common/rdtsc.h"
#include "common/ntapi.h"
#include "common/signal_context.h"
#include "core/libraries/libs.h"
#include "core/libraries/kernel/equeue.h"
#include "core/libraries/kernel/time.h"
#include "core/libraries/kernel/process.h"
#include "core/memory.h"
#include "core/signals.h"
#include "sdl_window.h"
#include "video_core/renderdoc.h"
#include "video_core/cache_storage.h"
#include "video_core/renderer_vulkan/vk_rasterizer.h"
#include "freeze_watchdog.h"
namespace iss::diagnostics {void RecordReadbackTime(u64 rip,u64 address,u64 elapsed,unsigned kind);}

Frontend::WindowSDL *g_window=nullptr;
namespace Libraries::GnmDriver {void RegisterLib(Core::Loader::SymbolsResolver*);}
namespace Libraries::VideoOut {void RegisterLib(Core::Loader::SymbolsResolver*);}
namespace Libraries::Ajm {void RegisterLib(Core::Loader::SymbolsResolver*);}
namespace Libraries::Videodec {void RegisterLib(Core::Loader::SymbolsResolver*);}
namespace Libraries::Kernel {void RegisterEventQueue(Core::Loader::SymbolsResolver*);}
namespace {
IssGpuCallbacks host{};
struct HostAliasMapping {u64 guest{}, size{}, physical{};};
thread_local std::vector<HostAliasMapping>* alias_visit_output{};
ReadbackHints readback_hints;
bool precise_reduction_site{};
bool precise_feedback_site{};
std::atomic<bool> explicit_readback_hints{};
bool FeedbackIdentityEnabled(){
    static const bool enabled=[]{const char* flag=std::getenv("ISS_FEEDBACK_IDENTITY");
        return flag&&flag[0]=='1';}();
    return enabled;
}
struct Symbol {std::string nid,library,module;uintptr_t pointer;};
std::vector<Symbol> symbols;
std::mutex mappings_mutex;
boost::icl::interval_set<VAddr> mappings;
std::thread window_thread;
std::mutex ready_mutex;
std::condition_variable ready_cv;
int ready=0;
/// ISS_THREAD_PRIORITY=1: the process runs above normal priority (as the emulator did) and the
/// threads that translate and present the graphics run one level above the game threads, so that
/// other programs and the game's own threads delay them less. Pipeline compilers stay low.
bool thread_priority_enabled(){static const bool on=[]{const char *p=std::getenv("ISS_THREAD_PRIORITY");return p&&p[0]=='1';}();return on;}
void raise_gpu_threads(){
    static const wchar_t *names[]={L"shadPS4:GpuCommandProcessor",L"shadPS4:GpuCommandRecorder",L"shadPS4:PresentThread",L"shadPS4:GpuUploadWorker",L"iss:DrawStageB"};
    const DWORD pid=GetCurrentProcessId();
    HANDLE snap=CreateToolhelp32Snapshot(TH32CS_SNAPTHREAD,0);if(snap==INVALID_HANDLE_VALUE)return;
    THREADENTRY32 te{};te.dwSize=sizeof(te);
    for(BOOL ok=Thread32First(snap,&te);ok;ok=Thread32Next(snap,&te)){
        if(te.th32OwnerProcessID!=pid)continue;
        HANDLE t=OpenThread(THREAD_SET_LIMITED_INFORMATION|THREAD_QUERY_LIMITED_INFORMATION,FALSE,te.th32ThreadID);if(!t)continue;
        PWSTR desc=nullptr;
        if(SUCCEEDED(GetThreadDescription(t,&desc))&&desc){
            for(const wchar_t *n:names)if(!wcscmp(desc,n)&&GetThreadPriority(t)<THREAD_PRIORITY_ABOVE_NORMAL){
                if(SetThreadPriority(t,THREAD_PRIORITY_ABOVE_NORMAL))std::fprintf(stderr,"PORT_TUNING thread_priority %ls=above_normal\n",desc);
            }
            LocalFree(desc);
        }
        CloseHandle(t);
    }
    CloseHandle(snap);
}
}
namespace Core::Loader {
void SymbolsResolver::AddSymbol(const char *nid,const char *library,const char *module,SymbolType,u64 address){symbols.push_back({nid,library,module,address});}
}
namespace Core {
class Emulator {
public:static void Configure(){auto& e=Common::ElfInfo::Instance();e.initialized=true;e.game_serial=host.serial;e.title=host.title;e.sdk_ver=host.sdk_version;e.psf_attributes.raw=host.psf_attributes;e.app_ver="01.00";}
};
void AddressSpace::Register(VAddr address,u64 size,u8 *backing){
    if(!size)return;
    std::scoped_lock guard{mappings_mutex};mappings.add(boost::icl::interval<VAddr>::right_open(address,address+size));
    for(u64 at=address;at<address+size;at+=16384)backing_pages[at>>14]=backing+(at-address);
}
boost::icl::interval_set<VAddr> AddressSpace::GetUsableRegions(){std::scoped_lock guard{mappings_mutex};return mappings;}
void AddressSpace::Protect(VAddr address,u64 size,MemoryPermission perms){
    if(!host.protect(address,size,static_cast<u32>(perms)))throw std::runtime_error("GPU protection failed");
}
void MemoryManager::SetRasterizer(Vulkan::Rasterizer *value){rasterizer=value;
    host.visit_mappings([](uint64_t address,uint64_t size,uint8_t *backing){iss_gpu_map(address,size,backing);});
}
u64 MemoryManager::ClampRangeSize(VAddr address,u64 size){return host.clamp(address,size);}
std::optional<MemoryManager::BackingSpan> MemoryManager::DescribeBacking(VAddr address,u64 size){
    if(size==0)return std::nullopt;
    const u64 contiguous=host.clamp(address,size);
    if(contiguous==0)return std::nullopt;
    u8* backing=host.backing(address,contiguous);
    if(!backing)return std::nullopt;
    return BackingSpan{reinterpret_cast<u64>(backing),contiguous};
}
std::optional<std::vector<MemoryManager::BackingAlias>>
MemoryManager::EnumerateBackingAliases(VAddr address,u64 size){
    const auto requested=DescribeBacking(address,size);
    if(!requested||requested->size!=size||
       requested->physical_start>std::numeric_limits<u64>::max()-size)return std::nullopt;
    std::vector<HostAliasMapping> mappings_snapshot;
    mappings_snapshot.reserve(4096); // native mapping table is bounded to 4096 entries
    auto* previous=alias_visit_output;
    alias_visit_output=&mappings_snapshot;
    host.visit_mappings([](u64 guest,u64 length,u8* physical){
        if(alias_visit_output)alias_visit_output->push_back(
            HostAliasMapping{guest,length,reinterpret_cast<u64>(physical)});
    });
    alias_visit_output=previous;
    const u64 physical_end=requested->physical_start+size;
    std::vector<BackingAlias> aliases;
    for(const auto& mapping:mappings_snapshot){
        if(mapping.size==0||mapping.guest>std::numeric_limits<u64>::max()-mapping.size||
           mapping.physical>std::numeric_limits<u64>::max()-mapping.size)return std::nullopt;
        const u64 lo=std::max(requested->physical_start,mapping.physical);
        const u64 hi=std::min(physical_end,mapping.physical+mapping.size);
        if(lo>=hi)continue;
        const u64 guest= mapping.guest+(lo-mapping.physical);
        // A CPU-only VA alias can still write the same physical bytes. The SRT provider
        // must arm it too, or reject the snapshot if PageManager cannot protect it.
        // The native table is append-only. A replaced VA or stale earlier entry must not
        // silently authorize a snapshot of the wrong physical bytes.
        if(host.backing(guest,hi-lo)!=reinterpret_cast<u8*>(lo))return std::nullopt;
        for(const auto& other:mappings_snapshot){
            if(&other==&mapping||other.size==0)continue;
            if(other.guest>std::numeric_limits<u64>::max()-other.size||
               other.physical>std::numeric_limits<u64>::max()-other.size)
                return std::nullopt;
            const u64 overlap_lo=std::max(guest,other.guest);
            const u64 overlap_hi=std::min(guest+(hi-lo),other.guest+other.size);
            if(overlap_lo<overlap_hi &&
               lo+(overlap_lo-guest)!=other.physical+(overlap_lo-other.guest))
                return std::nullopt;
        }
        aliases.push_back(BackingAlias{guest,lo,hi-lo});
        if(aliases.size()>64)return std::nullopt;
    }
    if(aliases.empty())return std::nullopt;
    std::sort(aliases.begin(),aliases.end(),[](const auto& a,const auto& b){
        return a.guest_address<b.guest_address;
    });
    aliases.erase(std::unique(aliases.begin(),aliases.end(),[](const auto& a,const auto& b){
        return a.guest_address==b.guest_address && a.size==b.size &&
               a.physical_start==b.physical_start;
    }),aliases.end());
    return aliases;
}
u64 MemoryManager::MappingEpoch() const noexcept {
    return host.mapping_epoch ? host.mapping_epoch() : 0;
}
bool MemoryManager::WithMappingReadLock(const std::function<bool(u64)>& publish) const {
    if (!host.mapping_lock_shared || !host.mapping_unlock_shared) return false;
    const u64 epoch=host.mapping_lock_shared();
    try {
        const bool accepted=publish(epoch);
        host.mapping_unlock_shared();
        return accepted;
    } catch (...) {
        host.mapping_unlock_shared();
        throw;
    }
}
u64 MemoryManager::PreferredReadbackWindow(VAddr address,u64 fallback){return precise_reduction_site||precise_feedback_site||explicit_readback_hints.load(std::memory_order_relaxed)?readback_hints.WindowFor(address,fallback):fallback;}
void MemoryManager::CopySparseMemory(VAddr address,u8 *dest,u64 size){
    static const bool bulk=[] {const char *p=std::getenv("ISS_BULK_COPY");return !p||p[0]!='0';}();
    if(!bulk){while(size){const u64 part=std::min<u64>(size,16384-(address&16383));
        if(auto *source=host.backing(address,part))std::memcpy(dest,source,part);else std::memset(dest,0,part);
        address+=part;dest+=part;size-=part;}return;}
    iss_copy_sparse(address,dest,size,host.backing);
}
u64 MemoryManager::HashSparseMemory(VAddr address,u64 size){std::vector<u8> bytes(size);CopySparseMemory(address,bytes.data(),size);return XXH3_64bits(bytes.data(),bytes.size());}
bool MemoryManager::TryWriteBacking(void *address,const void *data,u64 size){
    const auto guest=reinterpret_cast<uintptr_t>(address);
    auto *dest=host.backing(guest,size);
    if(!dest)return false;
    if(rasterizer)rasterizer->WithSrtDirectWrite([&]{std::memcpy(dest,data,size);});
    else std::memcpy(dest,data,size);
    return true;
}
bool MemoryManager::TryWriteBackingIf(void* address,const void* data,u64 size,
                                      u64 expected_native_epoch,
                                      const std::function<bool()>& predicate){
    if(!data||!size||!expected_native_epoch||!predicate||
       !host.mapping_lock_shared||!host.mapping_unlock_shared)return false;
    const auto guest=reinterpret_cast<uintptr_t>(address);
    // Resolve before taking the outer lock: host.backing takes its own shared
    // SRW lock and SRW locks do not promise recursive shared acquisition.
    auto* dest=host.backing(guest,size);
    if(!dest)return false;
    return iss::g4::ConditionalBackingWrite(
        [&](const auto& action){return WithMappingReadLock(action);},
        expected_native_epoch,predicate,[&]{
        if(rasterizer)rasterizer->WithSrtDirectWrite([&]{std::memcpy(dest,data,size);});
        else std::memcpy(dest,data,size);
    });
}
void MemoryManager::InvalidateMemory(VAddr address,u64 size){if(rasterizer)rasterizer->InvalidateMemory(address,size);}
}
namespace Libraries::Kernel {
boost::asio::io_context io_context;
static std::optional<boost::asio::executor_work_guard<boost::asio::io_context::executor_type>> work;
static std::thread service_thread;
void KernelSignalRequest(){} // Work guard keeps the timer executor alive.
int *PS4_SYSV_ABI __Error(){return host.errno_location();}
u64 PS4_SYSV_ABI sceKernelGetTscFrequency(){return host.tsc_frequency();}
u64 PS4_SYSV_ABI sceKernelReadTsc(){return Common::FencedRDTSC();}
u64 PS4_SYSV_ABI sceKernelGetProcessTime(){return host.process_time_us();}
u64 PS4_SYSV_ABI sceKernelGetProcessTimeCounter(){return host.process_time_us();}
u64 PS4_SYSV_ABI sceKernelGetProcessTimeCounterFrequency(){return 1000000;}
s32 PS4_SYSV_ABI sceKernelGetCompiledSdkVersion(s32 *out){if(!out)return static_cast<s32>(0x80020016);*out=host.sdk_version;return 0;}
s32 PS4_SYSV_ABI sceKernelIsNeoMode(){return 0;}
s32 PS4_SYSV_ABI sceKernelUsleep(u32 us){std::this_thread::sleep_for(std::chrono::microseconds(us));return 0;}
}
namespace Libraries::SystemService {bool IsSplashVisible(){return false;}}
namespace Core::PerfSampler {
void PollTuning(){iss::diagnostics::Poll();}
void RecordReadback(unsigned long long address,unsigned long long rip){
    static const bool trace=[] {const char *p=std::getenv("ISS_TRACE_READBACK");return p&&p[0]=='1';}();if(!trace)return;
    static std::mutex mutex;static std::map<unsigned long long,unsigned long long> counts;
    std::scoped_lock guard{mutex};if(counts.size()>=64&&!counts.contains(rip))return;
    auto count=++counts[rip];if(count<=4||(count&(count-1))==0)std::fprintf(stderr,"READBACK_SITE rip=0x%llx address=0x%llx count=%llu\n",rip,address,count);
}}
namespace VideoCore {
static std::atomic<u32> screenshot_requests,overlay_screenshot_requests;
void LoadRenderDoc(){}void StartCapture(){}void EndCapture(){}void TriggerCapture(){}
void SetOutputDir(const std::filesystem::path&,const std::string&){}
bool IsRenderDocLoaded(){return false;}
void RequestScreenshot(ScreenshotRequest request){if(request==ScreenshotRequest::GameOnly)++screenshot_requests;else if(request==ScreenshotRequest::WithOverlays)++overlay_screenshot_requests;}
u32 ConsumeGameOnlyScreenshotRequests(){return screenshot_requests.exchange(0);}
u32 ConsumeWithOverlaysScreenshotRequests(){return overlay_screenshot_requests.exchange(0);}
ScreenshotRequests ConsumeScreenshotRequests(){return {ConsumeGameOnlyScreenshotRequests(),ConsumeWithOverlaysScreenshotRequests()};}
}
namespace Common::FS {
static std::map<PathType,std::filesystem::path> paths;
const std::filesystem::path& GetUserPath(PathType type){return paths.at(type);}
std::string GetUserPathString(PathType type){return GetUserPath(type).string();}
void SetUserPath(PathType type,const std::filesystem::path& path){std::filesystem::create_directories(path);paths[type]=path;}
bool ValidatePath(const std::filesystem::path& path){return !path.empty();}
std::string PathToUTF8String(const std::filesystem::path& path){auto s=path.u8string();return std::string(s.begin(),s.end());}
}
extern "C" void iss_gpu_map(uint64_t address,uint64_t size,uint8_t *backing){
    if(FeedbackIdentityEnabled())
        iss::game::feedback::IdentityRegistry::Instance().Invalidate(address,size);
    auto *memory=Core::Memory::Instance();memory->GetAddressSpace().Register(address,size,backing);
    if(auto *r=memory->GetRasterizer()){r->RegisterMemory(address,size);r->MapMemory(address,size);}
}
extern "C" void iss_gpu_feedback_identity(uint64_t owner_object,uint64_t shader_object,
                                           uint64_t record_address){
    if(FeedbackIdentityEnabled()) {
        iss::game::feedback::IdentityRegistry::Instance().Publish(
            owner_object,shader_object,record_address);
        static std::atomic<uint64_t> events{};
        const auto count=++events;
        if(count<=8||(count&(count-1))==0)
            std::fprintf(stderr,"FB_IDENTITY_EBOOT count=%llu owner=0x%llx shader_object=0x%llx record=0x%llx\n",
                         static_cast<unsigned long long>(count),
                         static_cast<unsigned long long>(owner_object),
                         static_cast<unsigned long long>(shader_object),
                         static_cast<unsigned long long>(record_address));
    }
}
extern "C" int iss_gpu_readback(uint64_t address,uint64_t size){
    if(!size||!host.backing(address,size))return 0;
    auto *renderer=Core::Memory::Instance()->GetRasterizer();if(!renderer)return 0;
    readback_hints.Remember(address);explicit_readback_hints.store(true,std::memory_order_relaxed);
    const auto start=iss::diagnostics::Enabled()?Common::Perf::NowNs():0;
    renderer->ReadMemory(address,size,false,0);
    if(start)iss::diagnostics::RecordReadbackTime(0,address,Common::Perf::NowNs()-start,2);return 1;
}
extern "C" int iss_gpu_invalidate(uint64_t address,uint64_t size){
    // ISS_RUNTIME_WRITE_INVALIDATE=0 leaves file reads to their page faults (one per 4 KB page:
    // the game streams ~5 MB of textures every ~2 s, ~1470 exceptions, part of the periodic
    // 45 ms frames of 2026-10-17).
    static const bool enabled=[]{const char *p=std::getenv("ISS_RUNTIME_WRITE_INVALIDATE");return !p||p[0]!='0';}();
    if(!enabled||size<0x1000||!host.backing(address,size))return 0;
    auto *renderer=Core::Memory::Instance()->GetRasterizer();if(!renderer)return 0;
    return renderer->InvalidateMemory(address,size,false)?1:0;
}
extern "C" int iss_gpu_fault(void *context,void *address){
    if(precise_feedback_site&&!Common::IsWriteError(context)&&
       reinterpret_cast<uintptr_t>(Common::GetRip(context))==host.image_base+0x3f3eff)
        readback_hints.Remember(reinterpret_cast<uintptr_t>(address));
    // Register the hint before ReadMemory schedules its first copy/prefetch.
    if(precise_reduction_site&&!Common::IsWriteError(context)){
        const auto rip=reinterpret_cast<uintptr_t>(Common::GetRip(context));
        if(rip>=host.image_base){const auto offset=rip-host.image_base;
            if(offset==0x1717c0||offset==0x171800)readback_hints.Remember(reinterpret_cast<uintptr_t>(address));}}
    if(iss::repeat_census::enabled)[[unlikely]]
        iss::repeat_census::Note(iss::repeat_census::Fault,reinterpret_cast<uintptr_t>(address)&~uintptr_t{0xfff},0x1000,Common::IsWriteError(context)?1:0);
    const auto start=iss::diagnostics::Enabled()?Common::Perf::NowNs():0;
    const bool handled=Core::Signals::Instance()->DispatchAccessViolation(context,address);
    if(start&&handled)iss::diagnostics::RecordReadbackTime(reinterpret_cast<uintptr_t>(Common::GetRip(context)),reinterpret_cast<uintptr_t>(address),Common::Perf::NowNs()-start,Common::IsWriteError(context)?1:0);
    return handled;
}
extern "C" uintptr_t iss_gpu_resolve(const char *identity){
    std::string id(identity);std::vector<std::string> parts;size_t start=0;
    for(size_t end;(end=id.find('#',start))!=std::string::npos;start=end+1)parts.push_back(id.substr(start,end-start));parts.push_back(id.substr(start));
    if(parts.size()!=5||parts[2]!="1")return 0;
    for(const auto& s:symbols)if(s.nid==parts[0]&&s.library==parts[1]&&s.module==parts[3])return s.pointer;
    return 0;
}
// A short sleep that lasts what it asks (a high resolution waitable timer: ~0.5 ms, whatever
// the process timer resolution), for the renderer's polling waits.
extern "C" void IssPreciseSleepUs(unsigned us){
    thread_local HANDLE timer=CreateWaitableTimerExW(nullptr,nullptr,CREATE_WAITABLE_TIMER_HIGH_RESOLUTION,TIMER_ALL_ACCESS);
    if(!timer){SwitchToThread();return;}
    LARGE_INTEGER due;due.QuadPart=-static_cast<LONGLONG>(us)*10;
    if(SetWaitableTimer(timer,&due,0,nullptr,nullptr,FALSE))WaitForSingleObject(timer,INFINITE);
    else SwitchToThread();
}
extern "C" int iss_gpu_initialize(const IssGpuCallbacks *callbacks){
    if(!callbacks||callbacks->struct_size!=sizeof(host)||!callbacks->backing||!callbacks->clamp||!callbacks->visit_mappings||!callbacks->protect||!callbacks->mapping_epoch||!callbacks->mapping_lock_shared||!callbacks->mapping_unlock_shared)return -1;
    host=*callbacks;
    iss::timeline::Emit("gpu_initialize",0,0,host.image_base);
    iss::freeze_watchdog::SetImageBase(host.image_base);
    // The executing image (as native/engine_draw.cpp patches it), not a backing view.
    if(iss::native_shadow::Enabled())
        iss::native_shadow::Install(host.image_base,reinterpret_cast<const unsigned char*>(host.image_base+0x43D590));
    iss::native_gbuffer::InstallCensus(host.image_base);
    iss::native_gbuffer::Install(host.image_base);
    iss::pause_guard::Install(host.image_base);
    iss::native_passes::Install(host.image_base);
    Core::Emulator::Configure();
    if(const char *p=std::getenv("ISS_FEEDBACK_PAGE_READBACK");p&&p[0]=='1'){
        const unsigned char signature[]={0x8b,0x40,0x04,0x49,0x8b,0x8f,0xe8,0,0,0,0x89,0x41,0x08};
        const auto *code=host.backing(host.image_base+0x3f3eff,sizeof(signature));
        precise_feedback_site=code&&!std::memcmp(code,signature,sizeof(signature));
        std::fprintf(stderr,"FEEDBACK_READBACK verified=%d window=4096\n",precise_feedback_site);
    }
    if(const char *p=std::getenv("ISS_PRECISE_REDUCTION_READBACK");p&&p[0]=='1'){
        const unsigned char signature[]={0xc4,0xc1,0x7c,0x58,0x45,0xe0,0xc4,0xc1,0x74,0x58,0x4d,0x00};
        const auto *code=host.backing(host.image_base+0x1717c0,sizeof(signature));
        precise_reduction_site=code&&!std::memcmp(code,signature,sizeof(signature));
        std::fprintf(stderr,"REDUCTION_READBACK verified=%d window=4096 image_base=0x%llx\n",precise_reduction_site,(unsigned long long)host.image_base);
    }
    Common::NtApi::Initialize();
    for(int i=0;i<=static_cast<int>(Common::FS::PathType::LicensesDir);++i){
        auto type=static_cast<Common::FS::PathType>(i);Common::FS::SetUserPath(type,std::filesystem::path(host.user_dir)/("gpu-"+std::to_string(i)));}
    Common::FS::SetUserPath(Common::FS::PathType::UserDir,host.user_dir);
    // Isolate shader/driver caches for cold/warm comparisons without changing
    // the user's ordinary cache, saves or any other user directory.
    if (const char* cache_root = std::getenv("ISS_PIPELINE_CACHE_ROOT"); cache_root && *cache_root) {
        const auto path = std::filesystem::absolute(std::filesystem::u8path(cache_root));
        std::filesystem::create_directories(path);
        Common::FS::SetUserPath(Common::FS::PathType::CacheDir, path);
        std::fprintf(stderr, "PORT_TUNING pipeline_cache_root=%s\n", path.string().c_str());
    }
    Common::Perf::GetTuning().async_pipelines=0; // Never skip a draw while compiling its pipeline.
    const bool render_reuse=[] {const char *p=std::getenv("ISS_RENDER_REUSE");return p&&p[0]=='1';}();
    auto& tuning=Common::Perf::GetTuning();
    if(const char *p=std::getenv("ISS_PROBE_HOST_IMPORT");p&&p[0]=='1')tuning.probe_host_import=1;
    if(const char *p=std::getenv("ISS_NATIVE_FEEDBACK_PROBE");p&&p[0]=='1')tuning.probe_host_import=1;
    tuning.stream_dedup=render_reuse;tuning.stream_static=render_reuse;
    tuning.sync_memo=render_reuse;tuning.spec_memo=render_reuse;
    tuning.region_summary=render_reuse;tuning.tex_memo=render_reuse;
    tuning.skip_redundant_binds=render_reuse;tuning.shader_params_memo=render_reuse;
    tuning.skip_empty_draws=render_reuse;tuning.pipeline_key_memo=render_reuse;
    tuning.rt_memo=render_reuse;tuning.reuse_buffer_checks=render_reuse;
    const bool descriptor_fast=[] {const char *p=std::getenv("ISS_DESCRIPTOR_FAST_PATH");return p&&p[0]=='1';}();
    tuning.large_read_memo=descriptor_fast;
    std::fprintf(stderr,"PORT_TUNING large_read_memo=%d\n",descriptor_fast);
    const bool occlusion=[] {const char *p=std::getenv("ISS_REAL_OCCLUSION");return p&&p[0]=='1';}();
    tuning.real_occlusion=occlusion;
    const char *readback_setting=std::getenv("ISS_READBACK_WINDOW_KB");
    const long readback_kb=readback_setting?std::strtol(readback_setting,nullptr,10):512;
    tuning.readback_window_bytes=(readback_kb==64||readback_kb==128)?u64(readback_kb)*1024:512*1024;
    std::fprintf(stderr,"PORT_TUNING readback_window_bytes=%llu\n",(unsigned long long)tuning.readback_window_bytes.load());
    const bool parallel_render=[] {const char *p=std::getenv("ISS_PARALLEL_RENDER");return p&&p[0]=='1';}();
    tuning.threaded_recording=parallel_render;tuning.recorder_spin_us=parallel_render?20:0;tuning.readback_queue=parallel_render;
    const bool compute_fairness=[] {const char *p=std::getenv("ISS_COMPUTE_FAIRNESS");return p&&p[0]=='1';}();
    tuning.asc_yield_packets=compute_fairness?16:0;
    const bool async_compute=[] {const char *p=std::getenv("ISS_ASYNC_COMPUTE");return p&&p[0]=='1';}();
    tuning.asc_queue=async_compute;
    std::fprintf(stderr,"PORT_TUNING asc_queue=%d\n",async_compute);
    std::fprintf(stderr,"PORT_TUNING asc_yield_packets=%u\n",compute_fairness?16:0);
    std::fprintf(stderr,"PORT_TUNING threaded_recording=%d readback_queue=%d recorder_spin_us=%u\n",parallel_render,parallel_render,parallel_render?20:0);
    std::fprintf(stderr,"PORT_TUNING render_reuse=%d real_occlusion=%d\n",render_reuse,occlusion);
    const bool upload_worker=[] {const char *p=std::getenv("ISS_UPLOAD_WORKER");return p&&p[0]=='1';}();
    Common::Perf::GetTuning().async_uploads=upload_worker;
    std::fprintf(stderr,"PORT_TUNING async_uploads=%d async_pipelines=0\n",upload_worker);
    if(const char *p=std::getenv("ISS_GPU_PROFILE");p&&(p[0]=='1'||p[0]=='2'))Common::Perf::GetTuning().gpu_profile=p[0]-'0';
    {const char *p=std::getenv("ISS_COMPUTE_PHASES");iss::phase::g_enabled=p&&p[0]=='1';
     std::fprintf(stderr,"PORT_TUNING compute_phases=%d\n",iss::phase::Enabled()?1:0);}
    // ISS_TIMER_1MS=1 (off by default, the user's choice): 1 ms timer resolution for this
    // process. Without it a timed wait of 1-2 ms (the game's condition variables, event flags and
    // semaphores with a timeout, every Sleep(1)) lasts ~15.6 ms on Windows 11 (measured Sleep(1) =
    // 15.1 ms). Scripted city A/B (ab2, 2026-10-18 night, 2 runs each): no measurable difference.
    if(const char *p=std::getenv("ISS_TIMER_1MS");p&&p[0]=='1'){
        if(HMODULE winmm=LoadLibraryW(L"winmm.dll")){
            using TimeBeginPeriod=UINT(WINAPI*)(UINT);
            if(auto begin=reinterpret_cast<TimeBeginPeriod>(GetProcAddress(winmm,"timeBeginPeriod"))){
                const UINT result=begin(1);
                std::fprintf(stderr,"PORT_TUNING timer_1ms=%d\n",result==0?1:0);
            }
        }
    }
    if(thread_priority_enabled()){
        const BOOL ok=SetPriorityClass(GetCurrentProcess(),ABOVE_NORMAL_PRIORITY_CLASS);
        std::fprintf(stderr,"PORT_TUNING process_priority=above_normal ok=%d\n",ok?1:0);
    }
    // ISS_TUNING="name=value,...": existing renderer toggles for A/B tests, applied last.
    if(const char *p=std::getenv("ISS_TUNING")){
        const auto set=[](auto& field,const std::string& value){field=static_cast<std::remove_reference_t<decltype(field.load())>>(std::strtoull(value.c_str(),nullptr,0));};
        std::string all=p;size_t at=0;
        while(at<all.size()){
            size_t end=all.find(',',at);if(end==std::string::npos)end=all.size();
            const std::string item=all.substr(at,end-at);at=end+1;
            const size_t eq=item.find('=');if(eq==std::string::npos)continue;
            const std::string name=item.substr(0,eq),value=item.substr(eq+1);bool known=true;
            if(name=="late_protect")set(tuning.late_protect,value);
            else if(name=="parallel_protect")set(tuning.parallel_protect,value);
            else if(name=="bind_prefetch")set(tuning.bind_prefetch,value);
            else if(name=="scout")set(tuning.scout,value);
            else if(name=="scout_fast")set(tuning.scout_fast,value);
            else if(name=="hot_page_faults")set(tuning.hot_page_faults,value);
            else if(name=="async_written_uploads")set(tuning.async_written_uploads,value);
            else if(name=="async_uploads")set(tuning.async_uploads,value);
            else if(name=="readback_write_discard")set(tuning.readback_write_discard,value);
            else if(name=="readback_write_merge")set(tuning.readback_write_merge,value);
            else if(name=="gpu_threads_priority")set(tuning.gpu_threads_priority,value);
            else if(name=="asc_yield_packets")set(tuning.asc_yield_packets,value);
            else if(name=="recorder_spin_us")set(tuning.recorder_spin_us,value);
            else if(name=="tiered_compute_pipelines")set(tuning.tiered_compute_pipelines,value);
            // relaxed_site=<eboot offset> (repeatable, up to 8): reads of GPU written memory from
            // that 256 byte block of the game's code do not wait for the GPU (readback_relaxed_sites).
            else if(name=="relaxed_site"){const u32 n=tuning.num_relaxed_sites.load();if(n<Common::Perf::Tuning::MaxRelaxedSites){tuning.relaxed_sites[n]=std::strtoull(value.c_str(),nullptr,0)&~0xFFull;tuning.num_relaxed_sites.store(n+1);}}
            else if(name=="relaxed_all")tuning.relaxed_all=std::strtoull(value.c_str(),nullptr,0)!=0;
            else known=false;
            std::fprintf(stderr,"PORT_TUNING %s=%s%s\n",name.c_str(),value.c_str(),known?"":" (unknown, ignored)");
        }
    }
    window_thread=std::thread([]{
        try{
            g_window=new Frontend::WindowSDL(host.width,host.height,host.title);
            // The window answers from its first instant: the libraries, the renderer and the
            // shader preload start on their own thread while this one pumps the window's
            // messages (they ran here first: the window did not answer for 8 s, 30 s with the
            // launcher's shader cache, and Windows marked it "not responding"; winprobe1
            // 2026-10-18). The title shows the preload's progress meanwhile.
            static Core::Loader::SymbolsResolver resolver;
            std::atomic<int> init_state{0}; // 0 running, 1 done, -1 failed
            std::string init_error;
            std::thread init([&]{
                try{
                    Libraries::Kernel::RegisterEventQueue(&resolver);
                    Libraries::Kernel::work.emplace(boost::asio::make_work_guard(Libraries::Kernel::io_context));
                    Libraries::Kernel::service_thread=std::thread([]{Libraries::Kernel::io_context.run();});Libraries::Kernel::service_thread.detach();
                    Libraries::GnmDriver::RegisterLib(&resolver);Libraries::VideoOut::RegisterLib(&resolver);
                    Libraries::Ajm::RegisterLib(&resolver);
                    Libraries::Videodec::RegisterLib(&resolver);
                    init_state.store(1,std::memory_order_release);
                }catch(const std::exception& e){init_error=e.what();init_state.store(-1,std::memory_order_release);}
            });
            {
                auto last_title=std::chrono::steady_clock::now()-std::chrono::seconds(1);
                std::string shown;
                while(init_state.load(std::memory_order_acquire)==0){
                    if(!g_window->PollEvents()){
                        std::fputs("PORT_STOP window closed while loading\n",stderr);std::fflush(nullptr);
                        ::TerminateProcess(::GetCurrentProcess(),0);
                    }
                    const auto now=std::chrono::steady_clock::now();
                    if(now-last_title>std::chrono::milliseconds(40)){
                        last_title=now;
                        const unsigned entries=iss::g_preload_entries.load(std::memory_order_relaxed);
                        const unsigned total=iss::g_preload_total.load(std::memory_order_relaxed);
                        const int phase=iss::g_preload_phase.load(std::memory_order_relaxed);
                        // Phase 4: the first run compiles the game's compute shaders (ISS_AOT_WAIT).
                        std::string status=phase==0?"Starting...":
                            phase==1?"Loading shaders  "+std::to_string(entries)+(total?" / "+std::to_string(total):""):
                            phase==4?"Preparing shaders (first run only)  "+std::to_string(entries)+" / "+std::to_string(total):
                            phase==5?"Indexing the game's shaders (first run only)  "+std::to_string(entries)+" / "+std::to_string(total):
                            "Finishing shaders...";
                        const double fraction=(phase==1||phase==4||phase==5)&&total?double((std::min)(entries,total))/total:
                                              phase>=2?1.0:-1.0;
                        g_window->DrawLoading(fraction,status);
                        std::string title=std::string(host.title)+" - "+status;
                        if(title!=shown){SDL_SetWindowTitle(g_window->GetSDLWindow(),title.c_str());shown=title;}
                    }
                    std::this_thread::sleep_for(std::chrono::milliseconds(5));
                }
                init.join();
                SDL_SetWindowTitle(g_window->GetSDLWindow(),host.title);
                if(init_state.load()<0)throw std::runtime_error(init_error);
            }
            auto *renderer=Core::Memory::Instance()->GetRasterizer();
            const bool checkpoints=[] {const char *v=std::getenv("ISS_PIPELINE_CHECKPOINT");return !v||v[0]!='0';}();
            if(renderer&&checkpoints)renderer->GetPipelineCache().CheckpointDriverCache();
            {std::scoped_lock guard{ready_mutex};ready=1;}ready_cv.notify_all();
            const char *stop_path=std::getenv("ISS_STOP_FILE");
            auto last_checkpoint=std::chrono::steady_clock::now(),last_stop_check=last_checkpoint;
            auto last_priority=last_checkpoint-std::chrono::seconds(8);
            for(;;){
                bool stop=!g_window->PollEvents();auto now=std::chrono::steady_clock::now();
                if(stop_path&&now-last_stop_check>std::chrono::milliseconds(100)){std::error_code ec;stop|=std::filesystem::exists(stop_path,ec);last_stop_check=now;}
                if(stop){
                    if(renderer&&checkpoints)renderer->GetPipelineCache().CheckpointDriverCache();
                    // Our atomic checkpoint replaces the old destructor-only snapshot.
                    Storage::DataBase::Instance().SetBeforeClose({});Storage::DataBase::Instance().Close();
                    std::fputs("PORT_STOP cache writes drained; closing all guest threads and audio\n",stderr);std::fflush(nullptr);
                    ::TerminateProcess(::GetCurrentProcess(),0);return;
                }
                if(thread_priority_enabled()&&now-last_priority>std::chrono::seconds(10)){raise_gpu_threads();last_priority=now;}
                // Off the window thread: a changed driver cache takes ~120 ms to write.
                if(checkpoints&&renderer&&now-last_checkpoint>std::chrono::seconds(30)){
                    std::thread([renderer]{renderer->GetPipelineCache().CheckpointDriverCache();}).detach();last_checkpoint=now;}
                std::this_thread::sleep_for(std::chrono::milliseconds(2));
            }
        }catch(const std::exception& e){std::fprintf(stderr,"GPU initialization failed: %s\n",e.what());{std::scoped_lock guard{ready_mutex};ready=-1;}ready_cv.notify_all();}
    });window_thread.detach();
    std::unique_lock guard{ready_mutex};ready_cv.wait(guard,[]{return ready!=0;});return ready==1?0:-1;
}

// ISS_CP_CPU=L: the command processor thread limits the frame rate while several guest threads
// spin on rdtsc waiting for it. The physical core of logical processor L is taken out of the
// process default CPU sets (no other thread of the process runs there, its SMT sibling stays
// idle) and the command processor is given logical processor L alone. ISS_CP_CPU=auto, or an L
// this machine does not have (14 was tuned on 16 threads; a 12 thread CPU has no 14), takes the
// first logical processor of the last physical core. Unset, empty or "off": no pinning.
uint64_t IssImageBase() { return host.image_base; }

void IssPinCommandProcessor() {
    const char* value = std::getenv("ISS_CP_CPU");
    if (!value || !*value || !std::strcmp(value, "off")) return;
    const bool automatic = !std::strcmp(value, "auto");
    const long wanted = automatic ? -1 : std::strtol(value, nullptr, 10);
    ULONG length = 0;
    GetSystemCpuSetInformation(nullptr, 0, &length, GetCurrentProcess(), 0);
    std::vector<char> buffer(length);
    if (!length || !GetSystemCpuSetInformation(
            reinterpret_cast<PSYSTEM_CPU_SET_INFORMATION>(buffer.data()), length, &length,
            GetCurrentProcess(), 0)) {
        std::fprintf(stderr, "PORT_TUNING cp_cpu failed: no CPU set information\n");
        return;
    }
    struct Set { ULONG id; unsigned core; unsigned logical; unsigned group; };
    std::vector<Set> sets;
    for (size_t offset = 0; offset < length;) {
        const auto* info = reinterpret_cast<const SYSTEM_CPU_SET_INFORMATION*>(buffer.data() + offset);
        if (info->Type == CpuSetInformation) {
            sets.push_back({info->CpuSet.Id, info->CpuSet.CoreIndex,
                            info->CpuSet.LogicalProcessorIndex, info->CpuSet.Group});
        }
        offset += info->Size;
    }
    const Set* mine = nullptr;
    for (const auto& set : sets) {
        if (set.group == 0 && set.logical == wanted) mine = &set;
    }
    if (!mine) {
        for (const auto& set : sets) {
            if (set.group != 0) continue;
            if (!mine || set.core > mine->core || (set.core == mine->core && set.logical < mine->logical))
                mine = &set;
        }
        if (!mine) {
            std::fprintf(stderr, "PORT_TUNING cp_cpu failed: no logical processor in group 0\n");
            return;
        }
        if (!automatic)
            std::fprintf(stderr, "PORT_TUNING cp_cpu: logical processor %ld not found, using %u (last core)\n",
                         wanted, unsigned(mine->logical));
    }
    std::vector<ULONG> others;
    for (const auto& set : sets) {
        if (set.group != mine->group || set.core != mine->core) others.push_back(set.id);
    }
    const BOOL process_ok = !others.empty() &&
        SetProcessDefaultCpuSets(GetCurrentProcess(), others.data(), static_cast<ULONG>(others.size()));
    const ULONG id = mine->id;
    const BOOL thread_ok = SetThreadSelectedCpuSets(GetCurrentThread(), &id, 1);
    std::fprintf(stderr, "PORT_TUNING cp_cpu=%u%s core=%u process_default=%d(%zu sets) thread=%d\n",
                 unsigned(mine->logical), automatic ? " (auto)" : "", unsigned(mine->core), int(process_ok), others.size(), int(thread_ok));
}
