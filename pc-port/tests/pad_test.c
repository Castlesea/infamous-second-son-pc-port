// SPDX-License-Identifier: GPL-2.0-or-later
#include "runtime.h"
#include <SDL3/SDL.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#define CHECK(x) do{if(!(x)){fprintf(stderr,"PAD FAIL line %d: %s (%s)\n",__LINE__,#x,SDL_GetError());exit(1);}}while(0)
typedef int32_t(GUEST_ABI *Init)(void);
typedef int32_t(GUEST_ABI *Open)(int,int,int,const void*);
typedef int32_t(GUEST_ABI *Read)(int,void*);
static bool SDLCALL sensors(void *p,bool on){(void)p;(void)on;return true;}
static float scalar(const unsigned char *p,int offset){float f;memcpy(&f,p+offset,4);return f;}
void pad_spray_step(float q[4],float dx,float dy,float dt,int shake,double seconds,float angular[3],float accel[3]);
// The F7 spray can: held on its side (gravity along the controller's +X), a mouse move to the
// right turns it about the vertical (gyro -X), a move down about the horizontal (gyro +Y).
static void spray_test(void){
    float q[4]={0,0,.70710678f,.70710678f},w[3],a[3];
    pad_spray_step(q,0,0,.01f,0,0,w,a);CHECK(fabsf(a[0]-1)<.001f&&fabsf(a[1])<.001f&&fabsf(w[0])<.001f);
    pad_spray_step(q,10,0,.01f,0,0,w,a);CHECK(fabsf(w[0]+3)<.01f&&fabsf(w[1])<.01f&&fabsf(a[0]-1)<.001f);
    float norm=0;for(int i=0;i<4;++i)norm+=q[i]*q[i];CHECK(fabsf(norm-1)<.001f);
    pad_spray_step(q,0,10,.01f,0,0,w,a);CHECK(fabsf(w[1]-3)<.01f&&fabsf(w[0])<.01f&&a[0]<1);
    pad_spray_step(q,0,0,.01f,1,1.0/24,w,a);CHECK(a[0]>1.5f);
}
void pad_accel_gain(float accel[3],float gravity[3],float gain,float dt);
int pad_cursor_wakes(int dx,int dy,int pressed,unsigned long long still_ms);
// The cursor comes back only for real keyboard or mouse use: 6 px of motion or a press, once the
// controller has been still for half a second (not for input Windows counts on its own).
static void cursor_test(void){
    CHECK(!pad_cursor_wakes(0,0,0,10000));CHECK(!pad_cursor_wakes(3,4,0,10000));CHECK(!pad_cursor_wakes(-5,0,0,10000));
    CHECK(pad_cursor_wakes(6,0,0,10000));CHECK(pad_cursor_wakes(0,-6,0,10000));CHECK(pad_cursor_wakes(0,0,1,600));
    CHECK(!pad_cursor_wakes(50,0,1,400));CHECK(!pad_cursor_wakes(0,0,1,500));
}
// A Steam Controller (+-2 G) held on its side and shaken at 5 Hz clips at 2 G; with the gain the
// shake reaches past it as on a DualShock 4, and held still it reads gravity as measured.
static void accel_gain_test(void){
    float gravity[3],a[3]={1,0,0},peak=0,low=0;pad_accel_gain(a,gravity,2,0);CHECK(fabsf(a[0]-1)<.001f);
    for(int i=1;i<=200;++i){const float v=1+3*sinf(6.2831853f*5*i/250.f);a[0]=v>2?2:v<-2?-2:v;a[1]=a[2]=0;
        pad_accel_gain(a,gravity,2,1/250.f);if(i>100){if(a[0]>peak)peak=a[0];if(a[0]<low)low=a[0];}CHECK(a[0]<=4&&a[0]>=-4);}
    CHECK(peak>3&&low<-3);
    for(int i=0;i<500;++i){a[0]=1;a[1]=a[2]=0;pad_accel_gain(a,gravity,2,1/250.f);}CHECK(fabsf(a[0]-1)<.01f&&fabsf(a[1])<.001f);
}
int main(int argc,char **argv){
    int monitor=argc>1&&!strcmp(argv[1],"--monitor");
    // Keep the user's physical/PhonePad devices out of this synthetic contract test.
    if(!monitor){SDL_SetHint(SDL_HINT_JOYSTICK_HIDAPI,"0");SDL_SetHint(SDL_HINT_JOYSTICK_RAWINPUT,"0");
    SDL_SetHintWithPriority(SDL_HINT_JOYSTICK_HIDAPI_PS4,"0",SDL_HINT_OVERRIDE);
    SDL_SetHint(SDL_HINT_JOYSTICK_DIRECTINPUT,"0");SDL_SetHint(SDL_HINT_JOYSTICK_WGI,"0");SDL_SetHint("SDL_XINPUT_ENABLED","0");}
    if(!monitor)_putenv("ISS_PAD_MAP=guide=options");
    // The contract below reads right after each change: the inline path (ISS_PAD_THREAD=0). The
    // I/O thread serves the same snapshot, at most 2 ms old.
    if(!monitor)_putenv("ISS_PAD_THREAD=0");
    Init init=(Init)pad_resolve("hv1luiJrqQM#libScePad#1#libScePad#257");
    Open open=(Open)pad_resolve("xk0AcarP3V4#libScePad#1#libScePad#257");
    Read read=(Read)pad_resolve("YndgXqQVV7c#libScePad#1#libScePad#257");CHECK(init&&open&&read);CHECK(init()==0);
    if(monitor){CHECK(open(1,0,0,NULL)==1);unsigned char data[120];
        for(unsigned i=0;i<7500;++i){CHECK(read(1,data)==0);if(i%125==0){uint32_t buttons;memcpy(&buttons,data,4);
            printf("PAD_SAMPLE buttons=%x axes=%u,%u,%u,%u touches=%u gyro=%.3f,%.3f,%.3f accel=%.3f,%.3f,%.3f q=%.3f,%.3f,%.3f,%.3f\n",buttons,data[4],data[5],data[6],data[7],data[52],scalar(data,40),scalar(data,44),scalar(data,48),scalar(data,28),scalar(data,32),scalar(data,36),scalar(data,12),scalar(data,16),scalar(data,20),scalar(data,24));fflush(stdout);}SDL_Delay(4);}return 0;}
    SDL_VirtualJoystickDesc desc;SDL_INIT_INTERFACE(&desc);desc.type=SDL_JOYSTICK_TYPE_GAMEPAD;
    desc.naxes=6;desc.nbuttons=16;desc.axis_mask=63;desc.button_mask=65535;desc.name="PhonePad contract fixture";
    desc.vendor_id=0x054c;desc.product_id=0x09cc;
    SDL_VirtualJoystickTouchpadDesc touch={2,{0}};SDL_VirtualJoystickSensorDesc motion[2]={{SDL_SENSOR_GYRO,125},{SDL_SENSOR_ACCEL,125}};
    desc.ntouchpads=1;desc.touchpads=&touch;desc.nsensors=2;desc.sensors=motion;desc.SetSensorsEnabled=sensors;
    SDL_JoystickID id=SDL_AttachVirtualJoystick(&desc);CHECK(id);CHECK(SDL_IsGamepad(id));
    SDL_Joystick *joystick=SDL_OpenJoystick(id);CHECK(joystick);CHECK(open(1,0,0,NULL)==1);
    unsigned char data[120]={0};CHECK(read(1,data)==0);
    CHECK(SDL_SetJoystickVirtualAxis(joystick,0,32767));CHECK(SDL_SetJoystickVirtualAxis(joystick,1,-32768));
    CHECK(SDL_SetJoystickVirtualButton(joystick,SDL_GAMEPAD_BUTTON_SOUTH,true));
    CHECK(SDL_SetJoystickVirtualTouchpad(joystick,0,0,true,.25f,.75f,1));
    CHECK(SDL_SetJoystickVirtualTouchpad(joystick,0,1,true,.8f,.2f,1));
    float gyro[3]={0,0,1},accel[3]={SDL_STANDARD_GRAVITY,0,0};
    CHECK(SDL_SendJoystickVirtualSensorData(joystick,SDL_SENSOR_GYRO,SDL_GetTicksNS(),gyro,3));
    CHECK(SDL_SendJoystickVirtualSensorData(joystick,SDL_SENSOR_ACCEL,SDL_GetTicksNS(),accel,3));
    SDL_Delay(10);CHECK(read(1,data)==0);uint32_t buttons;memcpy(&buttons,data,4);
    CHECK(buttons&0x4000);CHECK(!(buttons&0x100000));CHECK(data[4]==255&&data[5]==0);
    CHECK(data[52]==2);CHECK(data[64]!=data[72]);
    uint16_t x,y;memcpy(&x,data+60,2);memcpy(&y,data+62,2);CHECK(x==479&&y==705);
    CHECK(fabsf(scalar(data,28)-1)<.001f);CHECK(fabsf(scalar(data,48)-1)<.001f);
    CHECK(scalar(data,20)>0);float norm=0;for(int i=12;i<=24;i+=4)norm+=scalar(data,i)*scalar(data,i);CHECK(fabsf(norm-1)<.001f);
    unsigned char contact=data[64];CHECK(SDL_SetJoystickVirtualTouchpad(joystick,0,0,true,.5f,.5f,1));CHECK(read(1,data)==0);CHECK(data[64]==contact);
    CHECK(SDL_SetJoystickVirtualTouchpad(joystick,0,0,false,0,0,0));CHECK(read(1,data)==0);CHECK(data[52]==1);
    CHECK(SDL_SetJoystickVirtualTouchpad(joystick,0,0,true,.5f,.5f,1));CHECK(read(1,data)==0);CHECK(data[52]==2&&data[64]!=contact);
    CHECK(SDL_SetJoystickVirtualTouchpad(joystick,0,0,false,0,0,0));CHECK(SDL_SetJoystickVirtualTouchpad(joystick,0,1,false,0,0,0));CHECK(read(1,data)==0);
    // A PlayStation controller keeps its own layout: Share does nothing, Options is Options.
    CHECK(SDL_SetJoystickVirtualButton(joystick,SDL_GAMEPAD_BUTTON_BACK,true));CHECK(read(1,data)==0);memcpy(&buttons,data,4);
    CHECK(data[52]==0&&!(buttons&0x100000)&&!(buttons&8));
    CHECK(SDL_SetJoystickVirtualButton(joystick,SDL_GAMEPAD_BUTTON_START,true));CHECK(read(1,data)==0);memcpy(&buttons,data,4);
    CHECK(data[52]==0&&!(buttons&0x100000)&&(buttons&8));
    CHECK(SDL_SetJoystickVirtualButton(joystick,SDL_GAMEPAD_BUTTON_BACK,false));CHECK(SDL_SetJoystickVirtualButton(joystick,SDL_GAMEPAD_BUTTON_START,false));CHECK(read(1,data)==0);CHECK(data[52]==0);
    SDL_CloseJoystick(joystick);CHECK(SDL_DetachVirtualJoystick(id));CHECK(read(1,data)==0);CHECK(data[52]==0&&scalar(data,48)==0);
    // A controller without a touchpad (Xbox) gets the layout: View presses the touchpad on both
    // sides, Menu is Options (recommended); View + D-pad left / right presses that side only. Guide
    // is Options here too (ISS_PAD_MAP, set before init).
    SDL_VirtualJoystickDesc xbox;SDL_INIT_INTERFACE(&xbox);xbox.type=SDL_JOYSTICK_TYPE_GAMEPAD;
    xbox.naxes=6;xbox.nbuttons=16;xbox.axis_mask=63;xbox.button_mask=65535;xbox.name="Xbox layout fixture";
    xbox.vendor_id=0x045e;xbox.product_id=0x02ff;
    id=SDL_AttachVirtualJoystick(&xbox);CHECK(id);joystick=SDL_OpenJoystick(id);CHECK(joystick);
    SDL_Delay(1100);CHECK(read(1,data)==0); // the next scan (once a second) picks it
    CHECK(SDL_SetJoystickVirtualButton(joystick,SDL_GAMEPAD_BUTTON_BACK,true));CHECK(read(1,data)==0);memcpy(&buttons,data,4);
    CHECK(data[52]==2&&(buttons&0x100000)&&!(buttons&8));memcpy(&x,data+60,2);CHECK(x==480);CHECK(data[64]!=data[72]);
    memcpy(&x,data+68,2);CHECK(x==1440);
    CHECK(SDL_SetJoystickVirtualButton(joystick,SDL_GAMEPAD_BUTTON_DPAD_RIGHT,true));CHECK(read(1,data)==0);memcpy(&buttons,data,4);
    memcpy(&x,data+60,2);CHECK(data[52]==1&&x==1440&&(buttons&0x100000)&&!(buttons&0x20));
    CHECK(SDL_SetJoystickVirtualButton(joystick,SDL_GAMEPAD_BUTTON_DPAD_RIGHT,false));
    CHECK(SDL_SetJoystickVirtualButton(joystick,SDL_GAMEPAD_BUTTON_DPAD_LEFT,true));CHECK(read(1,data)==0);memcpy(&buttons,data,4);
    memcpy(&x,data+60,2);CHECK(data[52]==1&&x==480&&!(buttons&0x80));
    CHECK(SDL_SetJoystickVirtualButton(joystick,SDL_GAMEPAD_BUTTON_DPAD_LEFT,false));
    CHECK(SDL_SetJoystickVirtualButton(joystick,SDL_GAMEPAD_BUTTON_START,true));CHECK(read(1,data)==0);memcpy(&buttons,data,4);
    CHECK(data[52]==2&&(buttons&0x100000)&&(buttons&8));
    CHECK(SDL_SetJoystickVirtualButton(joystick,SDL_GAMEPAD_BUTTON_BACK,false));CHECK(read(1,data)==0);memcpy(&buttons,data,4);
    CHECK(data[52]==0&&(buttons&8)&&!(buttons&0x100000));
    CHECK(SDL_SetJoystickVirtualButton(joystick,SDL_GAMEPAD_BUTTON_START,false));
    CHECK(SDL_SetJoystickVirtualButton(joystick,SDL_GAMEPAD_BUTTON_GUIDE,true));CHECK(read(1,data)==0);memcpy(&buttons,data,4);
    CHECK(data[52]==0&&(buttons&8)&&!(buttons&0x10000));
    CHECK(SDL_SetJoystickVirtualButton(joystick,SDL_GAMEPAD_BUTTON_GUIDE,false));
    // View held and the right stick pushed up: a swipe up from the bottom (no click, the camera
    // still); once it ends, View's own touch stays let go until View is released.
    CHECK(SDL_SetJoystickVirtualButton(joystick,SDL_GAMEPAD_BUTTON_BACK,true));CHECK(read(1,data)==0);CHECK(data[52]==2);
    CHECK(SDL_SetJoystickVirtualAxis(joystick,SDL_GAMEPAD_AXIS_RIGHTY,-32768));CHECK(read(1,data)==0);memcpy(&buttons,data,4);
    memcpy(&y,data+62,2);CHECK(data[52]==1&&y>600&&!(buttons&0x100000)&&data[6]==128&&data[7]==128);
    SDL_Delay(500);CHECK(read(1,data)==0);CHECK(data[52]==0);
    CHECK(SDL_SetJoystickVirtualAxis(joystick,SDL_GAMEPAD_AXIS_RIGHTY,0));CHECK(read(1,data)==0);CHECK(data[52]==0);
    CHECK(SDL_SetJoystickVirtualButton(joystick,SDL_GAMEPAD_BUTTON_BACK,false));CHECK(read(1,data)==0);
    CHECK(SDL_SetJoystickVirtualButton(joystick,SDL_GAMEPAD_BUTTON_BACK,true));CHECK(read(1,data)==0);CHECK(data[52]==2);
    CHECK(SDL_SetJoystickVirtualButton(joystick,SDL_GAMEPAD_BUTTON_BACK,false));
    // L3 + R3 (no gyro): the can gesture. The presses are not passed on; the controller turns on
    // its side (still upright right after the press), then shakes (the can's long axis, above 1.5 G),
    // then rests on its side; the right stick turns the can, A stays Cross (it no longer shakes).
    CHECK(SDL_SetJoystickVirtualButton(joystick,SDL_GAMEPAD_BUTTON_LEFT_STICK,true));
    CHECK(SDL_SetJoystickVirtualButton(joystick,SDL_GAMEPAD_BUTTON_RIGHT_STICK,true));CHECK(read(1,data)==0);memcpy(&buttons,data,4);
    CHECK(!(buttons&6)&&scalar(data,32)>.9f);
    CHECK(SDL_SetJoystickVirtualButton(joystick,SDL_GAMEPAD_BUTTON_LEFT_STICK,false));
    CHECK(SDL_SetJoystickVirtualButton(joystick,SDL_GAMEPAD_BUTTON_RIGHT_STICK,false));
    float peak=0;for(int i=0;i<60;++i){SDL_Delay(10);CHECK(read(1,data)==0);if(scalar(data,28)>peak)peak=scalar(data,28);}
    CHECK(peak>1.5f);
    SDL_Delay(1300);CHECK(read(1,data)==0);CHECK(fabsf(scalar(data,28)-1)<.01f);const float side=scalar(data,20);
    CHECK(SDL_SetJoystickVirtualAxis(joystick,SDL_GAMEPAD_AXIS_RIGHTX,32767));CHECK(read(1,data)==0);
    SDL_Delay(50);CHECK(read(1,data)==0);CHECK(data[6]==128&&scalar(data,40)<-.5f&&scalar(data,20)!=side);
    CHECK(SDL_SetJoystickVirtualAxis(joystick,SDL_GAMEPAD_AXIS_RIGHTX,0));
    CHECK(SDL_SetJoystickVirtualButton(joystick,SDL_GAMEPAD_BUTTON_SOUTH,true));CHECK(read(1,data)==0);memcpy(&buttons,data,4);
    CHECK(buttons&0x4000);CHECK(SDL_SetJoystickVirtualButton(joystick,SDL_GAMEPAD_BUTTON_SOUTH,false));
    // L3 + R3 again: shaken again, the can stays.
    CHECK(SDL_SetJoystickVirtualButton(joystick,SDL_GAMEPAD_BUTTON_LEFT_STICK,true));
    CHECK(SDL_SetJoystickVirtualButton(joystick,SDL_GAMEPAD_BUTTON_RIGHT_STICK,true));CHECK(read(1,data)==0);
    CHECK(SDL_SetJoystickVirtualButton(joystick,SDL_GAMEPAD_BUTTON_LEFT_STICK,false));
    CHECK(SDL_SetJoystickVirtualButton(joystick,SDL_GAMEPAD_BUTTON_RIGHT_STICK,false));
    peak=0;for(int i=0;i<12;++i){SDL_Delay(8);CHECK(read(1,data)==0);if(scalar(data,28)>peak)peak=scalar(data,28);}
    CHECK(peak>1.5f);
    // B (Circle) ends it and reaches the game: the stick is the camera again, the pad upright.
    CHECK(SDL_SetJoystickVirtualButton(joystick,SDL_GAMEPAD_BUTTON_EAST,true));CHECK(read(1,data)==0);memcpy(&buttons,data,4);
    CHECK(buttons&0x2000);CHECK(SDL_SetJoystickVirtualButton(joystick,SDL_GAMEPAD_BUTTON_EAST,false));
    CHECK(SDL_SetJoystickVirtualAxis(joystick,SDL_GAMEPAD_AXIS_RIGHTX,32767));CHECK(read(1,data)==0);
    CHECK(data[6]==255&&fabsf(scalar(data,24)-1)<.001f);
    // Walking Delsin (left stick past half for 0.5 s) ends it too: the graffiti is over.
    CHECK(SDL_SetJoystickVirtualAxis(joystick,SDL_GAMEPAD_AXIS_RIGHTX,0));
    CHECK(SDL_SetJoystickVirtualButton(joystick,SDL_GAMEPAD_BUTTON_LEFT_STICK,true));
    CHECK(SDL_SetJoystickVirtualButton(joystick,SDL_GAMEPAD_BUTTON_RIGHT_STICK,true));CHECK(read(1,data)==0);
    CHECK(SDL_SetJoystickVirtualButton(joystick,SDL_GAMEPAD_BUTTON_LEFT_STICK,false));
    CHECK(SDL_SetJoystickVirtualButton(joystick,SDL_GAMEPAD_BUTTON_RIGHT_STICK,false));
    CHECK(SDL_SetJoystickVirtualAxis(joystick,SDL_GAMEPAD_AXIS_RIGHTX,32767));CHECK(read(1,data)==0);CHECK(data[6]==128);
    CHECK(SDL_SetJoystickVirtualAxis(joystick,SDL_GAMEPAD_AXIS_LEFTY,-32768));CHECK(read(1,data)==0);
    SDL_Delay(600);CHECK(read(1,data)==0);CHECK(read(1,data)==0);CHECK(data[6]==255);
    CHECK(SDL_SetJoystickVirtualAxis(joystick,SDL_GAMEPAD_AXIS_LEFTY,0));CHECK(SDL_SetJoystickVirtualAxis(joystick,SDL_GAMEPAD_AXIS_RIGHTX,0));
    SDL_CloseJoystick(joystick);CHECK(SDL_DetachVirtualJoystick(id));
    spray_test();accel_gain_test();cursor_test();
    puts("PASS: SDL DS4 buttons, axes, two independent touch contacts, release, sensors, quaternion, disconnect, PlayStation and Xbox layouts, the F7 spray can, the accelerometer gain, the right stick swipes, View on both touchpad sides and one side with the D-pad, Menu as Options, the L3 + R3 can gesture (ends on B and on walking), and the cursor waking only for real mouse or key use.");
    return 0;
}
