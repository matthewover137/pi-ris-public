#ifndef __GETCODE_DANIEL__
#define __GETCODE_DANIEL__

#include "boot-defs-daniel.h"
#include "memmap.h"
#include "boot-crc32-daniel.h"
#include "gpio.h"

//blinks an led to debug
void debug_blinker(unsigned led) {
    gpio_set_output(led);
    while(1) {
        gpio_set_on(led);
        delay_cycles(1000000);
        gpio_set_off(led);
        delay_cycles(1000000);
    }
}

unsigned boot_get32() { 
    unsigned result;
    result = (unsigned)boot_get8();
    result |= (unsigned)(boot_get8() << 8);
    result |= (unsigned)(boot_get8() << 16);
    result |= (unsigned)(boot_get8() << 24);
    return result;
}

void boot_put32(unsigned val) { 
    boot_put8(val & 0xFF);
    boot_put8((val >> 8) & 0xFF);
    boot_put8((val >> 16) & 0xFF);
    boot_put8((val >> 24) & 0xFF);
}
//check for data for delay usec
unsigned data_received(unsigned delay) { 
    uint32_t start = timer_get_usec();
    while(1) {
        //check uart for data
        if (boot_has_data()) {
            return 1;
        }

        uint32_t curr = timer_get_usec();
        if ((curr - start) >= delay) {
            return 0;
        }
    }
}

//IMPORTANT: DOESNT WORK FOR NOW, NEEDS SOME UNIX CODE.
static void boot_putk(char *msg) {
    boot_put8(PRINT_STRING);

    int n = strlen(msg);
    boot_put32(n);
    for (int i = 0; msg[i]; i++) {
        boot_put8(msg[i]);
    }
}

enum {
    debug = 1, //0 off, 1 on 
    error1 = 17,
    error2,
    error3,
    error4,
};

uint32_t get_code() {
    
    //step 1 pi side bootloader.md
    //keep poll for PUT_PROG_INFO
    unsigned usec_delay = 300 * 1000; //300ms
    while(!data_received(usec_delay)) { 
        boot_put8(GET_PROG_INFO);
    }
    
    //step 3 pi side bootloader.md
    //ack prog info  
    if (boot_get8() != PUT_PROG_INFO) {
        panic("expected PUT_PROG_INFO as response to GET_PROG_INFO");
    }
    unsigned boot_addr = boot_get32();
    unsigned nbytes = boot_get32();
    unsigned received_crc = boot_get32();

    //check that binary will not interfere with current scheme
    unsigned scheme_start = (unsigned)PUT32;
    unsigned scheme_end = (unsigned)__code_end__;

    unsigned boot_end = boot_addr + nbytes;

    if (scheme_end >= boot_addr && boot_end >= scheme_start) { 
        boot_put8(BOOT_ERROR);
        if (debug) {
            debug_blinker(error1);
        }
        rpi_reboot(); 
    }
    boot_put8(GET_CODE);
    boot_put32(received_crc);

    //step 5 pi side bootloader.md
    //put code into place.
    if (boot_get8() != PUT_CODE) {
        boot_put8(BOOT_ERROR);
        if (debug) {
            debug_blinker(error2);
        }
        rpi_reboot();
    }

    for (int i = 0; i < nbytes; i++) {
        uint8_t byte = boot_get8();

        PUT8(boot_addr + i, byte);

    }

    unsigned code_crc = crc32((uint8_t*)boot_addr, nbytes);
    if (received_crc != code_crc) {
        boot_put8(BOOT_ERROR);
        if (debug) {
            debug_blinker(error3);
        }
        rpi_reboot();
    }
    //BUG: uart gets cooked here
    //very weird. BOOT SUCCESS is not recieved
    // and instead, 0 is recieved and nothing else (to my knowledge)
    // If I have the error4 debug thing on, boot success is recieved.
    // I have suspicions that it is non deterministic because 
    // hello.c in basic tests works with the blinking led always,
    // but I remember receiving boot success before and now it is 
    // working but I am getting 0 instead of boot success. wahoo

    //NOTE: turns out its something to do with uart init and uart destroy in lab 7,
    //which is where all my failing tests of this bootloader exist.

    //NOTE: WORKS IF THERE IS A DELAY HERE

    boot_put8(BOOT_SUCCESS);
    while (!((GET32(0x20215064) >> 9) & 1)); //JAMES CHEN SAID SO (clears uart tx buffer)
    // if (debug) {
    //     debug_blinker(error4);
    // }

    
    //branch handled in main, just return boot_addr
    return boot_addr;
}
#endif