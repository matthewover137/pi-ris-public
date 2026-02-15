# Inference server for Pi bare metal.
#
# Bakes msg_model.bin and tokenizer.bin directly into kernel.img
# using objcopy. No filesystem needed — GPU loads it all to 0x8000.

PROGS := main.c
COMMON_SRC := llama2.c

# Path to llama2.c data files
LLAMA_DIR := ../../llama2.c

# Model binary to bake in. Override with: make MODEL=path/to/model.bin
MODEL ?= $(LLAMA_DIR)/models/msg15M.bin

# The baked-in data files (objcopy'd into linkable .o) + kmalloc
PREBUILT_OBJS := msg_model.o tokenizer.o ../libpi/staff-objs/staff-kmalloc.o

# Don't try to bootload (this is meant to go on SD card as kernel.img)
RUN := 0

# Math library (for expf, sqrtf, sinf, cosf, powf) and libgcc (for float ABI helpers)
LIBM := $(shell arm-none-eabi-gcc -mcpu=arm1176jzf-s -print-file-name=libm.a)
LIBGCC := $(shell arm-none-eabi-gcc -mcpu=arm1176jzf-s -print-libgcc-file-name)
LIB_POST := $(LIBM) $(LIBGCC)

# Use hardware floating point (VFP) — massive speedup for inference
# -O3 for max optimization (loop unrolling, etc.)
OPT_LEVEL := -O3
CFLAGS_EXTRA := -mfloat-abi=softfp -mfpu=vfp -ffast-math -funroll-loops

# Use our local libpi, not the course one
CS140E_2026_PATH := ..
include ../libpi/mk/Makefile.robust-v2

# --- objcopy rules: convert raw binary data into linkable .o files ---
#
# objcopy creates symbols:
#   _binary_msg_model_bin_start  (pointer to first byte)
#   _binary_msg_model_bin_end    (pointer past last byte)
#   _binary_tokenizer_bin_start
#   _binary_tokenizer_bin_end
#
# We copy locally first so the symbol names are clean (no path chars).

msg_model.o: $(MODEL)
	@echo "BAKE  $(MODEL) ($$(du -h $< | cut -f1))"
	@cp $< msg_model.bin
	@$(ARM)-objcopy -I binary -O elf32-littlearm -B arm msg_model.bin $@
	@rm msg_model.bin

tokenizer.o: $(LLAMA_DIR)/tokenizer.bin
	@echo "BAKE  tokenizer.bin ($$(du -h $< | cut -f1))"
	@cp $< tokenizer.bin
	@$(ARM)-objcopy -I binary -O elf32-littlearm -B arm tokenizer.bin $@
	@rm tokenizer.bin

# --- final kernel image: just copy the .bin ---
all:: kernel.img

kernel.img: main.bin
	cp main.bin kernel.img
	@echo ""
	@echo "=== kernel.img ready ($$(du -h kernel.img | cut -f1)) ==="
	@echo "Copy to SD card root alongside bootcode.bin and start.elf"

clean::
	rm -f msg_model.o tokenizer.o kernel.img
