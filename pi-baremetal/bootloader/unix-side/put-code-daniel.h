#ifndef __PUT_CODE_DANIEL__
#define __PUT_CODE_DANIEL__
#include "libunix.h"
#include "boot-defs-daniel.h"

//required for my-install.c
enum { TRACE_FD = 21 };
enum { TRACE_CONTROL_ONLY = 1, TRACE_ALL = 2 };
extern int trace_p;

void simple_boot(int fd, unsigned boot_addr, const uint8_t *code, unsigned nbytes);

#endif