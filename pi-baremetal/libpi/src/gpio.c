/*
 * Implement the following routines to set GPIO pins to input or 
 * output, and to read (input) and write (output) them.
 *  1. DO NOT USE loads and stores directly: only use GET32 and 
 *    PUT32 to read and write memory.  See <start.S> for thier
 *    definitions.
 *  2. DO USE the minimum number of such calls.
 * (Both of these matter for the next lab.)
 *
 * See <rpi.h> in this directory for the definitions.
 *  - we use <gpio_panic> to try to catch errors.  For lab 2
 *    it only infinite loops since we don't have <printk>
 */
#include "rpi.h"
#include "gpio.h"

// See broadcomm documents for magic addresses and magic values.
//
// If you pass addresses as:
//  - pointers use put32/get32.
//  - integers: use PUT32/GET32.
//  semantics are the same.
enum {
    // Max gpio pin number.
    GPIO_MAX_PIN = 53,

    GPIO_BASE = 0x20200000,
    gpio_set0  = (GPIO_BASE + 0x1C),
    gpio_clr0  = (GPIO_BASE + 0x28),
    gpio_lev0  = (GPIO_BASE + 0x34)

    // <you will need other values from BCM2835!>
};

//
// Part 1 implement gpio_set_on, gpio_set_off, gpio_set_output
//

// set <pin> to be an output pin.
//
// NOTE: fsel0, fsel1, fsel2 are contiguous in memory, so you
// can (and should) use ptr calculations versus if-statements!
void gpio_set_output(unsigned pin) {
    gpio_set_function(pin, GPIO_FUNC_OUTPUT);
}

// Set GPIO <pin> = on.
// ASSUMPTION: GPSET0 and GPSET1 are CONTIGUOUS IN MEMORY
void gpio_set_on(unsigned pin) {
    if(pin > GPIO_MAX_PIN)
        gpio_panic("illegal pin=%d\n", pin);

    unsigned register_addr = gpio_set0 + ((pin >> 5) * 4); //0 or 1
    unsigned value = 1U << (pin & 0x1F);
    PUT32(register_addr, value);
}

// Set GPIO <pin> = off
// ASSUMPTION: GPCLR0 and GPCLR1 are CONTIGUOUS IN MEMORY
void gpio_set_off(unsigned pin) {
    if(pin > GPIO_MAX_PIN)
        gpio_panic("illegal pin=%d\n", pin);
    
    unsigned register_addr = gpio_clr0 + ((pin >> 5) * 4); //0 or 1
    unsigned value = 1U << (pin & 0x1F);
    PUT32(register_addr, value);
}

// Set <pin> to <v> (v \in {0,1})
void gpio_write(unsigned pin, unsigned v) {
    if(v)
        gpio_set_on(pin);
    else
        gpio_set_off(pin);
}

//
// Part 2: implement gpio_set_input and gpio_read
//

// set <pin> = input.
void gpio_set_input(unsigned pin) {
    gpio_set_function(pin, GPIO_FUNC_INPUT);
}

// Return 1 if <pin> is on, 0 if not.
int gpio_read(unsigned pin) {
    unsigned v = 0;

    if(pin > GPIO_MAX_PIN)
        gpio_panic("illegal pin=%d\n", pin);

    unsigned register_addr = gpio_lev0 + ((pin >> 5) * 4); //0 or 1
    unsigned mask = 1U << (pin & 0x1F); //all except the one we are reading

    unsigned value = GET32(register_addr);
    value &= mask;
    
    return value >> (pin & 0x1F);
}

void gpio_set_function(unsigned pin, gpio_func_t func) {
    if(pin > GPIO_MAX_PIN)
        gpio_panic("illegal pin=%d\n", pin);
    if((func & 0b111) != func)
        // NOTE: it says "func" not "function" and uses "%x" not "%d"
        gpio_panic("illegal func=%x\n", func); 

    unsigned register_addr = (pin / 10) * 4 + GPIO_BASE; // ASSUMPTION: map to 10s place because in groups of 10
    unsigned start_bit = pin % 10 * 3; // ASSUMPTION: see pg 93

    unsigned register_bits = GET32(register_addr);

    unsigned clear_mask = ~(0x7 << start_bit);
    register_bits &= clear_mask;

    unsigned set_out_bits = func << start_bit;
    register_bits |= set_out_bits;

    PUT32(register_addr, register_bits);
}
