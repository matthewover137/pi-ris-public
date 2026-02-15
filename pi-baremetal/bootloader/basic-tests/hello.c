#include "rpi.h"
#include "gpio.h"

void notmain() {
    printk("hello, world!\n");
    // uart_put8(0b00110011);
    // uart_put8(0b00000000);
    // enum { led = 21 };

    // gpio_set_output(led);
    // while(1) {
    //     gpio_set_on(led);
    //     delay_cycles(1000000);
    //     gpio_set_off(led);
    //     delay_cycles(1000000);
    // }
}