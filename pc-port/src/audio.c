// SPDX-License-Identifier: GPL-2.0-or-later
// Native audio output with per-channel conversion and real-time buffer pacing.
#include "runtime.h"
#include <SDL3/SDL.h>
#include <stdio.h>
#include <string.h>
#include <stdlib.h>
// type 4 (the PS4 pad speaker): out = 0 for the game's own layout, else the channels written to the
// controller's sound device (the sound in 1/2, the rest silent: a DualSense's 3/4 are its actuators).
// mix: an 8-channel port folded to stereo here (PS4 downmix, see downmix_mode); clips/clip_tick: its
// samples over full scale, logged every 30 s; peak/peak_tick: the pad speaker's level, logged every 5 s.
typedef struct {int active,channels,frames,floating,standard,type,out,mix;unsigned clips;float peak;int32_t volume[8];SDL_AudioStream *stream;SDL_AudioDeviceID dev;uint64_t next,clip_tick,peak_tick;} Port;
typedef struct {int32_t handle;const void *data;} Output;
static SRWLOCK lock=SRWLOCK_INIT;
static Port ports[25];static int initialized;
static int32_t err(unsigned n){return (int32_t)(UINT32_C(0x80260000)|n);}
static int32_t GUEST_ABI init(void){
    if(initialized)return err(14);if(!SDL_InitSubSystem(SDL_INIT_AUDIO)){fprintf(stderr,"Audio init: %s\n",SDL_GetError());return err(15);}initialized=1;return 0;
}
static Port *port(int handle){int i=handle&255;return (handle&0x3f000000)==0x20000000&&i<25&&ports[i].active?&ports[i]:NULL;}
// The PS4 pad speaker (port type 4: the phone, the spray can's rattle, draining a power source). On the
// console it plays on the controller; until 2026-11-07 it played on the PC's default output.
// ISS_PAD_SPEAKER=auto (default): a controller sound device ("Wireless Controller" / "DualSense") with 4+
//   channels (a DualSense, real on USB or DSZ's virtual one; its Quadraphonic setting), else the
//   default output; checked every 2 s, so a controller connected later takes over.
// ISS_PAD_SPEAKER=controller: as auto, also a 2-channel controller device. default: the PC's output.
//   off: not played. ISS_PAD_SPEAKER_VOLUME=<percent> (0-200, default 100) on whichever device.
enum {SPK_AUTO,SPK_CONTROLLER,SPK_DEFAULT,SPK_OFF};
static int spk_mode=-1,spk_volume=100;
static void spk_settings(void){
    if(spk_mode>=0)return;const char *m=getenv("ISS_PAD_SPEAKER"),*v=getenv("ISS_PAD_SPEAKER_VOLUME");
    spk_mode=!m||!*m||!_stricmp(m,"auto")?SPK_AUTO:!_stricmp(m,"controller")?SPK_CONTROLLER:!_stricmp(m,"off")||!strcmp(m,"0")?SPK_OFF:SPK_DEFAULT;
    if(v&&*v){spk_volume=atoi(v);if(spk_volume<0)spk_volume=0;if(spk_volume>200)spk_volume=200;}
    static const char *const names[]={"auto","controller","default","off"};
    printf("Audio pad speaker: %s, volume %d%%\n",names[spk_mode],spk_volume);
}
// The device the pad speaker should use now (0: none) and the channels to write (0: the game's own).
static SDL_AudioDeviceID spk_target(int *out){
    *out=0;if(spk_mode==SPK_OFF)return 0;if(spk_mode==SPK_DEFAULT)return SDL_AUDIO_DEVICE_DEFAULT_PLAYBACK;
    // Pass 0: "DualSense ..." (the pad itself on USB, or a USB dongle such as DS5Dongle); pass 1: any
    // controller device, such as DSZ's virtual "Wireless Controller" (Bluetooth pads have no device).
    int count=0;SDL_AudioDeviceID *ids=SDL_GetAudioPlaybackDevices(&count),found=0;
    for(int pass=0;pass<2&&!found;++pass)for(int i=0;ids&&i<count&&!found;++i){const char *name=SDL_GetAudioDeviceName(ids[i]);
        if(!name||(!strstr(name,"Wireless Controller")&&!strstr(name,"DualSense"))||(pass==0&&!strstr(name,"DualSense")))continue;
        SDL_AudioSpec spec;if(!SDL_GetAudioDeviceFormat(ids[i],&spec,NULL))continue;
        if(spec.channels>=4)*out=4;else if(spk_mode==SPK_CONTROLLER&&spec.channels>=1)*out=spec.channels>=2?2:1;else continue;
        found=ids[i];}
    SDL_free(ids);return found?found:SDL_AUDIO_DEVICE_DEFAULT_PLAYBACK;
}
// Opens p->stream on dev (the lock held). 0 on failure.
static int open_stream(Port *p,SDL_AudioDeviceID dev,int out){
    if(p->stream){SDL_DestroyAudioStream(p->stream);p->stream=NULL;}
    p->dev=dev;p->out=out;if(!dev)return 1;
    SDL_AudioSpec spec={SDL_AUDIO_F32,out?out:p->channels,48000};p->stream=SDL_OpenAudioDeviceStream(dev,&spec,NULL,NULL);
    if(!p->stream&&dev!=SDL_AUDIO_DEVICE_DEFAULT_PLAYBACK){fprintf(stderr,"Audio pad speaker open: %s, using the default output\n",SDL_GetError());
        p->dev=SDL_AUDIO_DEVICE_DEFAULT_PLAYBACK;p->out=0;spec.channels=p->channels;p->stream=SDL_OpenAudioDeviceStream(p->dev,&spec,NULL,NULL);}
    if(!p->stream)return 0;
    SDL_ResumeAudioStreamDevice(p->stream);
    if(p->type==4){const char *name=p->dev==SDL_AUDIO_DEVICE_DEFAULT_PLAYBACK?"default output":SDL_GetAudioDeviceName(p->dev);
        printf("Audio pad speaker -> %s (%d ch)\n",name?name:"?",p->out?p->out:p->channels);}
    return 1;
}
// ISS_AUDIO_DOWNMIX=ps4 (default): an 8-channel port on a stereo (or mono) default output is folded here
// as a PS4 does (and shadPS4's "PS4-accurate downmix"): L = FL + 0.7071 (FC + BL + SL), R likewise, LFE
// dropped. =sdl: SDL's own 7.1 -> stereo matrix, as before 2026-11-07 (front 0.21, about -13.5 dB).
static int downmix_mode=-1;
static int want_downmix(int channels){
    if(downmix_mode<0){const char *m=getenv("ISS_AUDIO_DOWNMIX");downmix_mode=!(m&&!_stricmp(m,"sdl"));
        printf("Audio downmix: %s\n",downmix_mode?"ps4":"sdl");}
    if(!downmix_mode||channels!=8)return 0;
    SDL_AudioSpec spec;return SDL_GetAudioDeviceFormat(SDL_AUDIO_DEVICE_DEFAULT_PLAYBACK,&spec,NULL)&&spec.channels>=1&&spec.channels<=2;
}
static int32_t GUEST_ABI open_audio(int user,int type,int index,uint32_t frames,uint32_t frequency,uint32_t param){
    (void)user;(void)index;if(!initialized)return err(15);if(!frames||frames>2048||(frames&255))return err(6);if(frequency!=48000)return err(8);
    unsigned format=param&255;if(format>7)return err(7);static const int channels[]={1,2,8,1,2,8,8,8};
    int first,last;switch(type){case 0:first=0;last=7;break;case 1:first=last=8;break;case 2:first=9;last=12;break;case 3:first=13;last=16;break;case 4:first=17;last=20;break;case 126:first=21;last=22;break;case 127:first=23;last=24;break;default:return err(10);}
    AcquireSRWLockExclusive(&lock);int id;for(id=first;id<=last&&ports[id].active;++id){}if(id>last){ReleaseSRWLockExclusive(&lock);return err(5);}
    Port *p=&ports[id];memset(p,0,sizeof(*p));p->channels=channels[format];p->frames=(int)frames;p->floating=format>=3&&format!=6;p->standard=format>=6;p->type=type;
    int out=0;SDL_AudioDeviceID dev=SDL_AUDIO_DEVICE_DEFAULT_PLAYBACK;if(type==4){spk_settings();dev=spk_target(&out);}
    else if(want_downmix(p->channels)){p->mix=1;out=2;printf("Audio port %d: PS4 stereo downmix\n",id);}
    if(!open_stream(p,dev,out)){fprintf(stderr,"Audio open: %s\n",SDL_GetError());ReleaseSRWLockExclusive(&lock);return err(3);}
    for(int i=0;i<8;++i)p->volume[i]=32768;p->active=1;ReleaseSRWLockExclusive(&lock);
    printf("Audio opened: %d channels, %u frames, 48000 Hz (type %d)\n",p->channels,frames,type);return 0x20000000|(type<<16)|id;
}
static int32_t output_one(int handle,const void *data,int pace){
    // The pad speaker follows the controller (device list read outside the lock, every 2 s).
    int want_out=0;SDL_AudioDeviceID want=0;int check=0;
    if(((handle>>16)&255)==4&&(spk_mode==SPK_AUTO||spk_mode==SPK_CONTROLLER)){static uint64_t checked;const uint64_t t=SDL_GetTicks();
        if(t-checked>=2000){checked=t;want=spk_target(&want_out);check=1;}}
    AcquireSRWLockExclusive(&lock);Port *p=port(handle);if(!p){ReleaseSRWLockExclusive(&lock);return err(1);}
    if(check&&(want!=p->dev||want_out!=p->out)&&!open_stream(p,want,want_out))fprintf(stderr,"Audio pad speaker: %s\n",SDL_GetError());
    uint64_t period=(uint64_t)p->frames*1000000000/48000,now=SDL_GetTicksNS();
    if(!p->next||now>p->next+8*period)p->next=now;
    uint64_t deadline=p->next;p->next+=period;
    float converted[2048*8];static const int remap[]={0,1,2,3,6,7,4,5};
    const float extra=p->type==4?(float)spk_volume/100.0f:1.0f;
    if(!p->stream){}
    else if(data&&p->type==4&&p->out){for(int f=0;f<p->frames;++f){float sum=0; // controller device: the mix in 1/2
            for(int c=0;c<p->channels;++c){int from=f*p->channels+c;
                float value=p->floating?((const float *)data)[from]:(float)((const int16_t *)data)[from]/32768.0f;
                sum+=value*(float)p->volume[c]/32768.0f;}
            sum=sum*extra/(float)p->channels;
            const float a=sum<0?-sum:sum;if(a>p->peak)p->peak=a;
            for(int c=0;c<p->out;++c)converted[f*p->out+c]=c<2?sum:0.0f;}
        if(!SDL_PutAudioStreamData(p->stream,converted,p->frames*p->out*4)){ReleaseSRWLockExclusive(&lock);return err(4);}
    }else if(data){for(int f=0;f<p->frames;++f)for(int c=0;c<p->channels;++c){int target=p->channels==8&&!p->standard?remap[c]:c;int from=f*p->channels+c,to=f*p->channels+target;
        float value=p->floating?((const float *)data)[from]:(float)((const int16_t *)data)[from]/32768.0f;
        converted[to]=value*(float)p->volume[c]/32768.0f*extra;}
        int put=p->channels;
        if(p->mix){const float k=0.7071f; // SDL order: FL FR FC LFE BL BR SL SR; in place (f*2 <= f*8)
            for(int f=0;f<p->frames;++f){const float *s=converted+f*8;
                const float l=s[0]+k*(s[2]+s[4]+s[6]),r=s[1]+k*(s[2]+s[5]+s[7]);
                if(l>1||l<-1)++p->clips;if(r>1||r<-1)++p->clips;
                converted[f*2]=l;converted[f*2+1]=r;}
            put=2;}
        if(p->type==4)for(int i=0;i<p->frames*p->channels;++i){const float a=converted[i]<0?-converted[i]:converted[i];if(a>p->peak)p->peak=a;}
        if(!SDL_PutAudioStreamData(p->stream,converted,p->frames*put*4)){ReleaseSRWLockExclusive(&lock);return err(4);}
    }else SDL_ClearAudioStream(p->stream);
    if(p->mix&&now-p->clip_tick>=30000000000ull){if(p->clip_tick&&p->clips)printf("Audio downmix 30s: %u samples over full scale\n",p->clips);p->clips=0;p->clip_tick=now;}
    if(p->type==4&&now-p->peak_tick>=5000000000ull){if(p->peak_tick&&p->peak>=0.01f)printf("Audio pad speaker 5s: peak %.2f -> %s\n",p->peak,p->out?"controller":p->dev?"default output":"off");p->peak=0;p->peak_tick=now;}
    ReleaseSRWLockExclusive(&lock);if(pace){now=SDL_GetTicksNS();if(now<deadline)SDL_DelayPrecise(deadline-now);}return 0;
}
static int32_t GUEST_ABI output_audio(int handle,const void *data){return output_one(handle,data,1);}
static int32_t GUEST_ABI outputs_audio(const Output *list,uint32_t count){if(!list||!count||count>25)return err(4);for(uint32_t i=0;i<count;++i){int32_t e=output_one(list[i].handle,list[i].data,i+1==count);if(e)return e;}return 0;}
static int32_t GUEST_ABI close_audio(int handle){AcquireSRWLockExclusive(&lock);Port *p=port(handle);if(!p){ReleaseSRWLockExclusive(&lock);return err(1);}if(p->stream)SDL_DestroyAudioStream(p->stream);p->stream=NULL;p->active=0;ReleaseSRWLockExclusive(&lock);return 0;}
static int32_t GUEST_ABI volume_audio(int handle,uint32_t mask,const int32_t *values){if(!values)return err(4);AcquireSRWLockExclusive(&lock);Port *p=port(handle);if(!p){ReleaseSRWLockExclusive(&lock);return err(1);}for(int i=0;i<p->channels;++i)if(mask&(1u<<i)){if(values[i]<0||values[i]>32768){ReleaseSRWLockExclusive(&lock);return err(9);}p->volume[i]=values[i];}ReleaseSRWLockExclusive(&lock);return 0;}
uintptr_t audio_resolve(const char *identity){
    if(!strstr(identity,"#libSceAudioOut#1#libSceAudioOut#257"))return 0;const char *api=runtime_api_name(identity);
#define API(n,f) if(!strcmp(api,n))return (uintptr_t)f
    API("sceAudioOutInit",init);API("sceAudioOutOpen",open_audio);API("sceAudioOutOutput",output_audio);API("sceAudioOutOutputs",outputs_audio);API("sceAudioOutClose",close_audio);API("sceAudioOutSetVolume",volume_audio);
#undef API
    return 0;
}
