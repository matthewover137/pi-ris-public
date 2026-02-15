PROGS := main-daniel.c
COMMON_SRC := 

# we have to override start --- look at it!
OUR_START := boot-start-daniel.S

# we can't bootload the bootloader.
RUN=0

# define this if you need to give the device for your pi
TTYUSB = 
BOOTLOADER = pi-install

include ./../../libpi/mk/Makefile.robust-v2

all:: kernel.img

clean::
	rm -f $(START) kernel.img

kernel.img: main-daniel.bin
	cp main-daniel.bin kernel.img
