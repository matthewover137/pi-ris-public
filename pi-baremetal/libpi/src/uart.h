#ifndef __UART__
#define __UART__

enum {
    AUX_IRQ = 0x20215000,
    AUX_ENB = AUX_IRQ + 0x4,
    AUX_IO = AUX_IRQ + 0x40,
    AUX_IER = AUX_IRQ + 0x44, 
    AUX_IIR = AUX_IRQ + 0x48,
    AUX_LCR = AUX_IRQ + 0x4C,
    AUX_MCR = AUX_IRQ + 0x50,
    AUX_LSR = AUX_IRQ + 0x54,
    AUX_MSR = AUX_IRQ + 0x58,
    AUX_SCRATCH = AUX_IRQ + 0x5C,
    AUX_CNTL = AUX_IRQ + 0x60,
    AUX_STAT = AUX_IRQ + 0x64,
    AUX_BAUD = AUX_IRQ + 0x68,
};

#endif