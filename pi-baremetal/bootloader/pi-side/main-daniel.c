//copied from main.c
#include "rpi.h"
#include "gpio.h"
/*******************************************************
 * UART implementation of our routines.
 */

// non-blocking: returns 1 if there is data, 0 otherwise.
static inline int boot_has_data(void) {
    return uart_has_data();
}

// returns 8-bits from the network connection.
static inline uint8_t boot_get8(void) {
    return uart_get8();
}

// sends 8-bits on the network connection.
static void boot_put8(uint8_t x) {
    uart_put8(x);
}

#include "get-code-daniel.h"

void notmain(void) {
   
    uint32_t addr = get_code();
    if(!addr)
        rpi_reboot();

    // blx to addr.  
    // could also call it as a function pointer.
    BRANCHTO(addr);
    not_reached();
}
