// SPDX-License-Identifier: GPL-2.0-or-later
#include "runtime.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "content_profile.h"
typedef struct { const void *buffer; uint32_t size,reserved; } PlayGoInit;
typedef struct { uint16_t chunk; int8_t locus,reserved; } ToDo;
typedef struct { uint64_t completed,total; } Progress;
static SRWLOCK lock=SRWLOCK_INIT;
static int loaded_playgo,playgo_initialized,playgo_opened,install_speed=1;
static volatile LONG loaded_modules[256];
static uint64_t speed_timestamp;
static uint64_t playgo_language_mask;
static int32_t GUEST_ABI pg_language(uint32_t h,uint64_t mask){
    if(h!=1||!playgo_opened)return (int32_t)UINT32_C(0x80b20009);
    AcquireSRWLockExclusive(&lock);playgo_language_mask=mask;ReleaseSRWLockExclusive(&lock);return 0;
}
static int local_user_initialized,login_pending;
static int content_initialized;
static int32_t GUEST_ABI content_init(const unsigned char *init,unsigned char *boot){
    if(!init||!boot)return (int32_t)UINT32_C(0x80d90002);
    for(unsigned i=0;i<32;++i)if(init[i])return (int32_t)UINT32_C(0x80d90002);
    memset(boot,0,40);content_initialized=1;return 0;
}
// Add-on content: the game's DLC are entitlement-only packages (its costumes and the Cole's Legacy
// mission ship in the base game; a license unlocks them). ISS_ADDCONT_LABELS="LABEL,LABEL,..." (the
// launcher reads them from the user's .pkg files, content id EP9000-CUSA00004_00-<label> or UP9000-CUSA00223_00-<label>) are
// reported owned; any other label has no entitlement. The base game folder is never touched.
static int32_t GUEST_ABI content_addon(uint32_t service,const void *label,void *info){
    if(!content_initialized||service||!label||!info)return (int32_t)UINT32_C(0x80d90002);
    char want[17];memcpy(want,label,16);want[16]=0;
    const char *list=getenv("ISS_ADDCONT_LABELS");
    for(const char *p=list;p&&*p;){
        const char *end=strchr(p,',');size_t n=end?(size_t)(end-p):strlen(p);
        if(n==16&&!memcmp(p,want,16)){
            // SceAppContentAddcontInfo: the label (17 bytes and padding to 20), then the status.
            unsigned char *out=info;memset(out,0,24);memcpy(out,want,16);
            const char *status=getenv("ISS_ADDCONT_STATUS");uint32_t s=status?(uint32_t)atoi(status):4u; // installed
            memcpy(out+20,&s,4);
            fprintf(stderr,"ADDCONT %s owned (status %u)\n",want,s);return 0;
        }
        p=end?end+1:NULL;
    }
    fprintf(stderr,"ADDCONT %s not owned\n",want);
    return (int32_t)UINT32_C(0x80d90007); // SCE_APP_CONTENT_ERROR_DRM_NO_ENTITLEMENT
}
typedef void (GUEST_ABI *NetCallback)(int event,void *data);
static struct { NetCallback fn; void *data; } net_callbacks[8];
static int net_control_initialized;
static int32_t GUEST_ABI netctl_init(void){net_control_initialized=1;return 0;}
static void GUEST_ABI netctl_term(void){net_control_initialized=0;memset(net_callbacks,0,sizeof(net_callbacks));}
static int32_t GUEST_ABI netctl_state(int *out){if(!out)return (int32_t)UINT32_C(0x80412107);*out=0;return 0;}
static int32_t GUEST_ABI netctl_info(int code,void *out){if(code<1||code>22||!out)return (int32_t)UINT32_C(0x80412107);return (int32_t)UINT32_C(0x80412108);}
static int32_t GUEST_ABI netctl_register(NetCallback fn,void *data,int *id){
    if(!fn||!id)return (int32_t)UINT32_C(0x80412107);AcquireSRWLockExclusive(&lock);
    for(int i=0;i<8;++i)if(!net_callbacks[i].fn){net_callbacks[i].fn=fn;net_callbacks[i].data=data;*id=i+1;ReleaseSRWLockExclusive(&lock);return 0;}
    ReleaseSRWLockExclusive(&lock);return (int32_t)UINT32_C(0x80412105);
}
static int32_t GUEST_ABI netctl_unregister(int id){if(id<1||id>8)return (int32_t)UINT32_C(0x80412107);AcquireSRWLockExclusive(&lock);net_callbacks[id-1].fn=NULL;ReleaseSRWLockExclusive(&lock);return 0;}
static int32_t GUEST_ABI user_initialize(const void *p){(void)p;local_user_initialized=1;login_pending=1;return 0;}
static int32_t GUEST_ABI user_terminate(void){local_user_initialized=login_pending=0;return 0;}
static int32_t GUEST_ABI user_initial(int32_t *id){if(!id)return (int32_t)UINT32_C(0x80960005);if(!local_user_initialized)return (int32_t)UINT32_C(0x80960002);*id=1;return 0;}
static int32_t GUEST_ABI user_list(int32_t *ids){if(!ids)return (int32_t)UINT32_C(0x80960005);ids[0]=1;ids[1]=ids[2]=ids[3]=-1;return 0;}
static int32_t GUEST_ABI user_event(int32_t *e){if(!e)return (int32_t)UINT32_C(0x80960005);if(!InterlockedExchange((volatile LONG *)&login_pending,0))return (int32_t)UINT32_C(0x80960007);e[0]=0;e[1]=1;return 0;}
// The system language (SCE_SYSTEM_SERVICE_PARAM_ID_LANG): ISS_LANGUAGE=<PS4 language id, 0-30>,
// set by the launcher from the languages of the game's package (CUSA00004 or CUSA00223); 17 when unset.
static int32_t system_language(void){
    const char *s=getenv("ISS_LANGUAGE");char *end;long id=s?strtol(s,&end,10):-1;
    return s&&*s&&!*end&&id>=0&&id<=30?(int32_t)id:17;
}
static int32_t GUEST_ABI system_param(int id,int32_t *out){
    if(!out)return (int32_t)UINT32_C(0x80a10003);
    switch(id){case 1:*out=system_language();break;case 2:*out=1;break;case 3:*out=1;break;
    case 4:{TIME_ZONE_INFORMATION tz;DWORD result=GetTimeZoneInformation(&tz);*out=-tz.Bias-(result==TIME_ZONE_ID_DAYLIGHT?tz.DaylightBias:tz.StandardBias);break;}
    case 5:{TIME_ZONE_INFORMATION tz;*out=GetTimeZoneInformation(&tz)==TIME_ZONE_ID_DAYLIGHT;break;}
    case 7:*out=0;break;case 1000:*out=1;break;default:return (int32_t)UINT32_C(0x80a10003);}return 0;
}
static int32_t GUEST_ABI system_status(void *p){if(!p)return (int32_t)UINT32_C(0x80a10003);memset(p,0,12);return 0;}
static int32_t GUEST_ABI system_string(int id,char *out,uint64_t size){
    if(id!=6||!out||!size||size>UINT32_MAX)return (int32_t)UINT32_C(0x80a10003);
    DWORD length=(DWORD)size;if(!GetComputerNameA(out,&length))return (int32_t)UINT32_C(0x80a10003);return 0;
}
static int32_t GUEST_ABI system_event(void *p){if(!p)return (int32_t)UINT32_C(0x80a10003);return (int32_t)UINT32_C(0x80a10004);}
static int32_t GUEST_ABI system_safe_area(float *ratio){if(!ratio)return (int32_t)UINT32_C(0x80a10003);*ratio=1.0f;return 0;}
static int32_t GUEST_ABI hide_splash(void){puts("System splash dismissed");return 0;}
// Local unpacked files need no optical-disc placement hint. This narrow contract
// follows shadPS4 DiscMap Func_7C980FFB0AA27E7A; data I/O still uses real host files.
static int32_t GUEST_ABI disc_local_hint(const char *path,int64_t offset,int64_t bytes,int *flags,int *a,int *b){
    if(!path||offset<0||bytes<0||!flags||!a||!b)return (int32_t)UINT32_C(0x80020016);
    *flags=*a=*b=0;return 0;
}
static int32_t GUEST_ABI disc_hdd(const char *path,int64_t offset,int64_t bytes,int *out){
    if(!path||offset<0||bytes<0||!out)return (int32_t)UINT32_C(0x81100001);
    return (int32_t)UINT32_C(0x81100004); // Unpacked files have no optical-package bitmap.
}
static int32_t pg_error(unsigned n){return (int32_t)(UINT32_C(0x80b20000)|n);}
static int32_t GUEST_ABI module_load(uint16_t id){
    if(!id)return (int32_t)UINT32_C(0x805a1000);
    if(id==0x83&&LOCAL_PACKAGE_COMPLETE){loaded_playgo=1;InterlockedExchange(&loaded_modules[id],1);puts("Loaded PlayGo backend: local package manifest complete");return 0;}
    if(id==0xa4){InterlockedExchange(&loaded_modules[id],1);puts("Loaded native Windows MsgDialog backend");return 0;}
    if(id==0xd7){InterlockedExchange(&loaded_modules[id],1);puts("Loaded DiscMap backend for unpacked local files");return 0;}
    if(id==0xa0){InterlockedExchange(&loaded_modules[id],1);puts("Loaded isolated SaveData backend");return 0;}
    if(id==0xad){InterlockedExchange(&loaded_modules[id],1);puts("Loaded local trophy-state backend");return 0;}
    if(id==0x8e&&gpu_resolve("qkgRiwHyheU#libSceVideodec#1#libSceVideodec#257")){InterlockedExchange(&loaded_modules[id],1);return 0;}
    if(id==0xb4){InterlockedExchange(&loaded_modules[id],1);puts("Loaded base-game AppContent backend");return 0;}
    printf("Module backend unavailable: id=0x%x\n",id);return (int32_t)UINT32_C(0x805a1001);
}
static int32_t GUEST_ABI module_loaded(uint16_t id){return id<256&&InterlockedCompareExchange(&loaded_modules[id],0,0)?0:(int32_t)UINT32_C(0x805a1001);}
static int32_t GUEST_ABI pg_initialize(const PlayGoInit *p){
    if(!p||!p->buffer)return pg_error(10);if(p->size<0x200000)return pg_error(11);
    AcquireSRWLockExclusive(&lock);
    int32_t r=playgo_initialized?pg_error(6):!LOCAL_PACKAGE_COMPLETE?pg_error(14):0;
    if(!r)playgo_initialized=1;ReleaseSRWLockExclusive(&lock);return r;
}
static int32_t GUEST_ABI pg_open(uint32_t *out,const void *p){
    if(!out)return pg_error(10);if(p)return pg_error(4);if(!playgo_initialized)return pg_error(5);
    *out=1;playgo_opened=1;return 0;
}
static int32_t GUEST_ABI pg_close(uint32_t handle){if(handle!=1)return pg_error(9);playgo_opened=0;return 0;}
static int32_t GUEST_ABI pg_terminate(void){playgo_opened=playgo_initialized=0;return 0;}
static int32_t valid(uint32_t handle,const uint16_t *ids,uint32_t count){
    if(!playgo_initialized)return pg_error(5);if(handle!=1||!playgo_opened)return pg_error(9);
    if(!ids)return pg_error(10);if(!count)return pg_error(11);
    for(uint32_t i=0;i<count;++i)if(ids[i]>=LOCAL_CHUNKS)return pg_error(12);return 0;
}
static int32_t GUEST_ABI pg_locus(uint32_t h,const uint16_t *ids,uint32_t count,int8_t *out){
    int32_t r=valid(h,ids,count);if(r)return r;if(!out)return pg_error(10);
    for(uint32_t i=0;i<count;++i)out[i]=3;return 0;
}
static int32_t GUEST_ABI pg_progress(uint32_t h,const uint16_t *ids,uint32_t count,Progress *out){
    int32_t r=valid(h,ids,count);if(r)return r;if(!out)return pg_error(10);
    out->completed=out->total=0;for(uint32_t i=0;i<count;++i)out->total+=chunk_sizes[ids[i]];
    out->completed=out->total;return 0;
}
static int32_t GUEST_ABI pg_eta(uint32_t h,const uint16_t *ids,uint32_t count,int64_t *out){
    int32_t r=valid(h,ids,count);if(r)return r;if(!out)return pg_error(10);*out=0;return 0;
}
static int32_t GUEST_ABI pg_todo(uint32_t h,ToDo *out,uint32_t count,uint32_t *entries){
    if(h!=1||!playgo_opened)return pg_error(9);if(!out||!entries)return pg_error(10);
    if(!count)return pg_error(11);*entries=0;return 0;
}
static int32_t GUEST_ABI pg_set_todo(uint32_t h,const ToDo *list,uint32_t count){
    if(h!=1||!playgo_opened)return pg_error(9);if(!list&&count)return pg_error(10);
    for(uint32_t i=0;i<count;++i){if(list[i].chunk>=LOCAL_CHUNKS)return pg_error(12);
        if(list[i].locus!=0&&list[i].locus!=2&&list[i].locus!=3)return pg_error(16);}return 0;
}
static int32_t GUEST_ABI pg_set_speed(uint32_t h,int speed){
    if(h!=1||!playgo_opened)return pg_error(9);if(speed<0||speed>2)return pg_error(13);
    AcquireSRWLockExclusive(&lock);install_speed=speed;speed_timestamp=GetTickCount64();ReleaseSRWLockExclusive(&lock);return 0;
}
static int32_t GUEST_ABI pg_speed(uint32_t h,int *speed){
    if(h!=1||!playgo_opened)return pg_error(9);if(!speed)return pg_error(10);
    AcquireSRWLockExclusive(&lock);if(!install_speed&&GetTickCount64()-speed_timestamp>30000)install_speed=1;
    *speed=install_speed;ReleaseSRWLockExclusive(&lock);return 0;
}
uintptr_t service_resolve(const char *identity){
    if(!strcmp(identity,"fJgP+wqifno#libSceDiscMap#1#libSceDiscMap#257"))return (uintptr_t)disc_local_hint;
    if(!strcmp(identity,"lbQKqsERhtE#libSceDiscMap#1#libSceDiscMap#257"))return (uintptr_t)disc_hdd;
    const char *api=runtime_api_name(identity);
    if(strstr(identity,"#libSceAppContent#1#libSceAppContentUtil#257")){
        if(!strcmp(api,"sceAppContentInitialize"))return (uintptr_t)content_init;
        if(!strcmp(api,"sceAppContentGetAddcontInfo"))return (uintptr_t)content_addon;
    }
    if(strstr(identity,"#libSceNetCtl#1#libSceNetCtl#257")){
        if(!strcmp(api,"sceNetCtlInit"))return (uintptr_t)netctl_init;
        if(!strcmp(api,"sceNetCtlTerm"))return (uintptr_t)netctl_term;
        if(!strcmp(api,"sceNetCtlGetState"))return (uintptr_t)netctl_state;
        if(!strcmp(api,"sceNetCtlGetInfo"))return (uintptr_t)netctl_info;
        if(!strcmp(api,"sceNetCtlRegisterCallback"))return (uintptr_t)netctl_register;
        if(!strcmp(api,"sceNetCtlUnregisterCallback"))return (uintptr_t)netctl_unregister;
    }
    if(strstr(identity,"#libSceSystemService#1#libSceSystemService#257")){
        if(!strcmp(api,"sceSystemServiceParamGetInt"))return (uintptr_t)system_param;
        if(!strcmp(api,"sceSystemServiceParamGetString"))return (uintptr_t)system_string;
        if(!strcmp(api,"sceSystemServiceGetStatus"))return (uintptr_t)system_status;
        if(!strcmp(api,"sceSystemServiceReceiveEvent"))return (uintptr_t)system_event;
        if(!strcmp(api,"sceSystemServiceGetDisplaySafeAreaInfo"))return (uintptr_t)system_safe_area;
        if(!strcmp(api,"sceSystemServiceHideSplashScreen"))return (uintptr_t)hide_splash;
    }
    if(strstr(identity,"#libSceUserService#1#libSceUserService#257")){
        if(!strcmp(api,"sceUserServiceInitialize"))return (uintptr_t)user_initialize;
        if(!strcmp(api,"sceUserServiceTerminate"))return (uintptr_t)user_terminate;
        if(!strcmp(api,"sceUserServiceGetInitialUser"))return (uintptr_t)user_initial;
        if(!strcmp(api,"sceUserServiceGetLoginUserIdList"))return (uintptr_t)user_list;
        if(!strcmp(api,"sceUserServiceGetEvent"))return (uintptr_t)user_event;
    }
    if(strstr(identity,"#libSceSysmodule#1#libSceSysmodule#257")){
        if(!strcmp(api,"sceSysmoduleLoadModule"))return (uintptr_t)module_load;
        if(!strcmp(api,"sceSysmoduleIsLoaded"))return (uintptr_t)module_loaded;
    }
    if(!strstr(identity,"#libScePlayGo#1#libScePlayGo#257") &&
       !strstr(identity,"#libScePlayGo#1#libScePlayGo#256"))return 0;
#define API(n,f) if(!strcmp(api,n))return (uintptr_t)f
    API("scePlayGoInitialize",pg_initialize);API("scePlayGoOpen",pg_open);API("scePlayGoClose",pg_close);
    API("scePlayGoTerminate",pg_terminate);API("scePlayGoGetLocus",pg_locus);API("scePlayGoGetProgress",pg_progress);
    API("scePlayGoGetEta",pg_eta);API("scePlayGoGetToDoList",pg_todo);API("scePlayGoSetToDoList",pg_set_todo);
    API("scePlayGoGetInstallSpeed",pg_speed);API("scePlayGoSetInstallSpeed",pg_set_speed);
    API("scePlayGoSetLanguageMask",pg_language);
#undef API
    return 0;
}
