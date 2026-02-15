// engler, cs140e: your code to find the tty-usb device on your laptop.
#include <assert.h>
#include <fcntl.h>
#include <string.h>
#include <sys/stat.h>

#include "libunix.h"

#define _SVID_SOURCE
#include <dirent.h>
static const char *ttyusb_prefixes[] = {
    "ttyUSB",	// linux
    "ttyACM",   // linux
    "cu.SLAB_USB", // mac os
    "cu.usbserial", // mac os
    // if your system uses another name, add it.
	0
};

static int filter(const struct dirent *d) {
    // scan through the prefixes, returning 1 when you find a match.
    // 0 if there is no match.
    for (int i = 0; ttyusb_prefixes[i] != 0; i++) {
        unsigned len = strlen(ttyusb_prefixes[i]);

        if (strncmp(d->d_name, ttyusb_prefixes[i], len) == 0) {
            return 1;
        }
    }
    return 0;
}

// find the TTY-usb device (if any) by using <scandir> to search for
// a device with a prefix given by <ttyusb_prefixes> in /dev
// returns:
//  - device name.
// error: panic's if 0 or more than 1 devices.
char *find_ttyusb(void) {
    // use <alphasort> in <scandir>
    // return a malloc'd name so doesn't corrupt.
    struct dirent **devices;
    int n = scandir("/dev", &devices, filter, alphasort);
    
    if (n != 1) {
        for (int i = 0; i < n; i++) {
            free(devices[i]);
        }
        free(devices);
        panic("found %d devices, expected exactly 1\n", n);
    }
    
    char *name;
    asprintf(&name, "/dev/%s", devices[0]->d_name);  

    //free
    free(devices[0]);
    free(devices);
    
    return name;
}

// return the most recently mounted ttyusb (the one
// mounted last).  use the modification time 
// returned by state.
char *find_ttyusb_last(void) {
    struct dirent **devices;
    int n = scandir("/dev", &devices, filter, alphasort);

    if (n == 0) {
        panic("found no devices. are you sure pi is connected?");
    }

    int max_index = -1;
    unsigned max_time = 0;
    char path[128];
    struct stat s;

    for (int i = 0; i < n; i++) {
        sprintf(path, "/dev/%s", devices[i]->d_name);
        stat(path, &s);
        if (s.st_mtime > max_time) {
            max_index = i;
            max_time = s.st_mtime;
        }
    }

    char *name;
    asprintf(&name, "/dev/%s", devices[max_index]->d_name);  

    //free
    for (int i = 0; i < n; i++) {
        free(devices[i]);
    }
    free(devices);
    
    return name; 
}

// return the oldest mounted ttyusb (the one mounted
// "first") --- use the modification returned by
// stat()
char *find_ttyusb_first(void) {
    struct dirent **devices;
    int n = scandir("/dev", &devices, filter, alphasort);

    if (n == 0) {
        panic("found no devices. are you sure pi is connected?");
    }

    int min_index = -1;
    unsigned min_time = -1; //large
    char path[128];
    struct stat s;

    for (int i = 0; i < n; i++) {
        sprintf(path, "/dev/%s", devices[i]->d_name);
        stat(path, &s);
        if (s.st_mtime < min_time) {
            min_index = i;
            min_time = s.st_mtime;
        }
    }

    char *name;
    asprintf(&name, "/dev/%s", devices[min_index]->d_name);  

    //free
    for (int i = 0; i < n; i++) {
        free(devices[i]);
    }
    free(devices);
    
    return name; 
}
