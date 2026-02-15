#include "put-code-daniel.h"

int trace_p = 0;

//IMPORTANT: NEEDS SOME FIXES TO MAKE get 32 wrapped as well
uint8_t get_opcode(int fd) {
    uint8_t opcode = get_uint8(fd);
    while(1) {
        if (opcode != PRINT_STRING){ 
            return opcode;
        }
        
        int n = get_uint32(fd);
        debug_output("ITS ME: putk %d bytes.\n", n);

        for (int i = 0; i < n; i++) {
            output("%c", get_uint8(fd));
        }
    }
}


void simple_boot(int fd, unsigned boot_addr, const uint8_t *code, unsigned nbytes) {
    //step 2 unix side bootloader.md
    //send over high level execution details
    debug_output("BOOTLOADER: PUTTING CODE PROFILE\n");
    uint8_t opcode;
    while((opcode = get_uint8(fd)) != GET_PROG_INFO);   
    //debug_output("ITS ME: PUT_PROG_INFO\n");
    put_uint8(fd, PUT_PROG_INFO);
    put_uint32(fd, ARMBASE);
    put_uint32(fd, nbytes);
    unsigned crc = our_crc32(code, nbytes);
    put_uint32(fd, crc);

    //step 4 unix side bootloader.md
    //check that pi knows its about to recieve code
    debug_output("BOOTLOADER: PUTTING CODE\n");
    while((opcode = get_uint8(fd)) == GET_PROG_INFO); //get rid of all of these
    if (opcode == GET_CODE) {
        unsigned received_crc = get_uint32(fd);
        if (crc != received_crc) {
            panic("unix side: incorrect crc returned. Expected: %d. Got: %d.\n", crc, received_crc);
        }

        put_uint8(fd, PUT_CODE);
        for (int i = 0; i < nbytes; i++) {
            put_uint8(fd, code[i]);
        }
    }
    else if (opcode == BOOT_ERROR) {
        panic("got BOOT_ERROR opcode, expected GET_CODE\n");
    }
    else {
        panic("unknown opcode received, expected GET_CODE or BOOT_ERROR\n");
    }

    //step 6 unix side bootloader.md
    //start echoing stuff to terminal (handled in my-install)
    opcode = get_uint8(fd);
    if (opcode == BOOT_ERROR) {
        panic("got BOOT_ERROR opcode, expected BOOT_SUCCESS.\n");
    }
    else if (opcode != BOOT_SUCCESS) {
        debug_output("ERROR: got %x not BOOT_SUCCESS\n", opcode);
        while (1) {
            opcode = get_uint8(fd);
            debug_output("ERROR: got %x not BOOT_SUCCESS\n", opcode);
        }
        panic("unknown opcode received, expected BOOT_SUCCESS.\n");
        
    }

    debug_output("BOOTLOADER: FINISHED\n");
}