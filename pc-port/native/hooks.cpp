// SPDX-License-Identifier: GPL-2.0-or-later
// Install only before original module initializers / guest entry. No live patching.
#include "reduction.h"
#include "feedback_identity.h"
#include "geometry.h"
#include "ultrawide.h"
#include "late_reads.h"
#include "../src/runtime.h"
#include <atomic>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <intrin.h>
namespace {
iss::native::ReductionBindings bindings{};
std::atomic<unsigned long long> calls{},fallbacks{};
const float *ReadResult(std::uint64_t address,std::size_t size){
    if(!gpu_readback(address,size))++fallbacks;
    // Keep guest-page protection active if another GPU write arrives after synchronization.
    return reinterpret_cast<const float*>(static_cast<std::uintptr_t>(address));
}
std::uint8_t ISS_SYSV NativeReduction(const void *image_info,std::uint64_t argument,void *context,float *output){
    const auto result=iss::native::Reduce(bindings,image_info,argument,context,output);
    const auto n=++calls;if(n==1||(n&(n-1))==0)std::fprintf(stderr,"NATIVE_REDUCTION calls=%llu available=%u fallback_reads=%llu\n",n,unsigned(result),fallbacks.load());
    return result;
}
std::uint64_t Hash(const unsigned char *p,std::size_t size){std::uint64_t h=14695981039346656037ULL;while(size--)h=(h^*p++)*1099511628211ULL;return h;}
// ISS_SPIN_PROBE=1: eboot+0x53b2b0 is the game's short sleep (below 501 us it spins on rdtsc, else
// sceKernelUsleep). The polling loops that wait for GPU labels (ring space, frame fences) call it.
// Replaced by the same behaviour plus, per caller, the time spent in it (diagnostic).
struct SpinSite{std::atomic<std::uint64_t> caller{},cycles{},calls{};};
SpinSite spin_sites[64];
std::atomic<std::uint64_t> spin_report_tsc{};
unsigned char *spin_image{};
std::uint64_t spin_frequency{};
__declspec(noinline) void SpinRecord(std::uint64_t caller,std::uint64_t cycles){
    const std::uint64_t key=caller-reinterpret_cast<std::uintptr_t>(spin_image);
    std::size_t slot=(key*0x9E3779B97F4A7C15ULL)>>58;
    for(std::size_t n=0;n<64;++n,slot=(slot+1)&63){
        auto &site=spin_sites[slot];std::uint64_t expected=0;
        if(site.caller.load(std::memory_order_relaxed)==key||site.caller.compare_exchange_strong(expected,key)||expected==key){
            site.cycles.fetch_add(cycles,std::memory_order_relaxed);site.calls.fetch_add(1,std::memory_order_relaxed);break;}
    }
    const std::uint64_t now=__rdtsc();std::uint64_t last=spin_report_tsc.load(std::memory_order_relaxed);
    if(now-last<spin_frequency*10||!spin_report_tsc.compare_exchange_strong(last,now))return;
    // Per 10 s window: milliseconds per second spent waiting, by call site.
    const double seconds=double(now-last)/double(spin_frequency);
    std::fprintf(stderr,"SPIN_PROBE window=%.1fs",seconds);
    for(auto &site:spin_sites){
        const std::uint64_t c=site.cycles.exchange(0),k=site.calls.exchange(0);
        if(k)std::fprintf(stderr," [0x%llx %.1fms/s %llu]",(unsigned long long)site.caller.load(),
            1000.0*double(c)/double(spin_frequency)/seconds,(unsigned long long)k);
    }
    std::fputc(10,stderr);
}
std::uint64_t ISS_SYSV SpinSleep(std::uint32_t us){
    const std::uint64_t caller=reinterpret_cast<std::uintptr_t>(__builtin_return_address(0));
    const std::uint64_t start=__rdtsc();
    if(us>=501){
        HANDLE timer=CreateWaitableTimerExW(nullptr,nullptr,0x2,TIMER_ALL_ACCESS);
        if(!timer)timer=CreateWaitableTimerW(nullptr,TRUE,nullptr);
        if(timer){LARGE_INTEGER due;due.QuadPart=-static_cast<std::int64_t>(us)*10;
            if(SetWaitableTimer(timer,&due,0,nullptr,nullptr,FALSE))WaitForSingleObject(timer,INFINITE);CloseHandle(timer);}
    }else{
        const std::uint64_t end=start+spin_frequency/1000000*us;
        while(__rdtsc()<end){}
    }
    const std::uint64_t stop=__rdtsc();
    SpinRecord(caller,stop-start);
    return stop;
}
int InstallSpinProbe(unsigned char *image,std::uint64_t size){
    const char *option=std::getenv("ISS_SPIN_PROBE");if(!option||option[0]!='1')return 0;
    constexpr std::size_t entry=0x53b2b0,length=0xa7;
    if(size<=entry+length||Hash(image+entry,length)!=0x89bfe4b7f4c237fbULL){std::fputs("SPIN_PROBE rejected: eboot function identity mismatch\n",stderr);return -1;}
    spin_image=image;spin_frequency=runtime_tsc_frequency();spin_report_tsc=__rdtsc();
    unsigned char jump[13]={0x48,0xb8};const auto target=reinterpret_cast<std::uintptr_t>(&SpinSleep);
    std::memcpy(jump+2,&target,8);jump[10]=0xff;jump[11]=0xe0;jump[12]=0x90;
    DWORD old,ignored;if(!VirtualProtect(image+entry,sizeof(jump),PAGE_EXECUTE_READWRITE,&old))return -1;
    std::memcpy(image+entry,jump,sizeof(jump));
    if(!VirtualProtect(image+entry,sizeof(jump),old,&ignored)||!FlushInstructionCache(GetCurrentProcess(),image+entry,sizeof(jump)))return -1;
    std::fputs("SPIN_PROBE installed at eboot+0x53b2b0\n",stderr);return 1;
}
int InstallReduction(unsigned char *image,std::uint64_t size){
    const char *option=std::getenv("ISS_NATIVE_REDUCTION");if(!option||option[0]!='1')return 0;
    constexpr std::size_t entry=0x171610,length=644;
    if(size<=0x1d0cbd4||Hash(image+entry,length)!=0x41698c67351a07aeULL){std::fputs("NATIVE_HOOK rejected: eboot function identity mismatch\n",stderr);return -1;}
    bindings.reset=image+0x1d0cbd4;bindings.frames=reinterpret_cast<std::uint32_t*>(image+0xe5e8c0);
    bindings.current=reinterpret_cast<std::uintptr_t*>(image+0xe5e908);bindings.next=reinterpret_cast<std::uintptr_t*>(image+0xe5e910);
    bindings.dispatch_state=image+0xe5e7c0;
    bindings.enter=reinterpret_cast<decltype(bindings.enter)>(image+0x11f6f0);
    bindings.leave=reinterpret_cast<decltype(bindings.leave)>(image+0x11f700);
    bindings.dispatch=reinterpret_cast<decltype(bindings.dispatch)>(image+0x1713e0);
    bindings.exponential=reinterpret_cast<decltype(bindings.exponential)>(image+0x984530);
    bindings.read=ReadResult;
    // Thirteen complete, non-PC-relative prologue bytes. RAX is caller-saved.
    unsigned char jump[13]={0x48,0xb8};const auto target=reinterpret_cast<std::uintptr_t>(&NativeReduction);
    std::memcpy(jump+2,&target,8);jump[10]=0xff;jump[11]=0xe0;jump[12]=0x90;
    DWORD old,ignored;if(!VirtualProtect(image+entry,sizeof(jump),PAGE_EXECUTE_READWRITE,&old))return -1;
    std::memcpy(image+entry,jump,sizeof(jump));
    if(!VirtualProtect(image+entry,sizeof(jump),old,&ignored)||!FlushInstructionCache(GetCurrentProcess(),image+entry,sizeof(jump)))return -1;
    std::fputs("NATIVE_HOOK installed reduction at eboot+0x171610; original producer and expf retained\n",stderr);return 1;
}
}
extern "C" int native_install_hooks(unsigned char *image,std::uint64_t size){
    const int reduction=InstallReduction(image,size);
    if(reduction<0)return reduction;
    const int geometry=iss::native::InstallIndexedGeometry(image,size);
    if(geometry<0)return geometry;
    const int spin=InstallSpinProbe(image,size);
    if(spin<0)return spin;
    const int markers=iss::native::InstallEngineMarkers(image,size);
    if(markers<0)return markers;
    const int jobs=iss::native::InstallRenderJobCensus(image,size);
    if(jobs<0)return jobs;
    const int ultrawide=iss::native::InstallUltrawide(image,size);
    if(ultrawide<0)return ultrawide;
    const int late_particles=iss::native::InstallParticleLateRead(image,size);
    if(late_particles<0)return late_particles;
    const int installed=reduction+geometry+spin+markers+jobs+ultrawide+late_particles;
    const char *feedback=std::getenv("ISS_FEEDBACK_IDENTITY");
    if(!feedback||feedback[0]!='1')return installed;
    if(!gpu_feedback_identity_available()){
        std::fputs("FEEDBACK_IDENTITY disabled: GPU DLL callback unavailable\n",stderr);
        return installed;
    }
    return iss::native::InstallFeedbackIdentityCall(image,size,gpu_feedback_identity)?installed+1:installed;
}
