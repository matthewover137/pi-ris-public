#include <assert.h>
#include <fcntl.h>
#include <sys/stat.h>
#include <unistd.h>

#include "libunix.h"

// allocate buffer, read entire file into it, return it.   
// buffer is zero padded to a multiple of 4.
//
//  - <size> = exact nbytes of file.
//  - for allocation: round up allocated size to 4-byte multiple, pad
//    buffer with 0s. 
//
// fatal error: open/read of <name> fails.
//   - make sure to check all system calls for errors.
//   - make sure to close the file descriptor (this will
//     matter for later labs).
// 
void *read_file(unsigned *size, const char *name) {
    // How: 
    //    - use stat() to get the size of the file.
    //    - round up to a multiple of 4.
    //    - allocate a buffer
    //    - zero pads to a multiple of 4.
    //    - read entire file into buffer (read_exact())
    //    - fclose() the file descriptor
    //    - make sure any padding bytes have zeros.
    //    - return it.   
    struct stat info;
    if (stat(name, &info) < 0) 
    panic("stat failed. name: %s\n", name);
    
    unsigned file_size = info.st_size;

    char *buf = calloc((file_size + 3) & ~0x3, sizeof(char));
    if (!buf) panic("calloc failed.\n");

    int fd = open(name, O_RDONLY);
    if (fd < 0) 
    panic("open failed. name: %s\n", name);

    if (file_size > 0)
    read_exact(fd, buf, file_size);
        
    *size = file_size;

    if (close(fd) < 0)
    panic("close failed.\n");
    return buf;
}
