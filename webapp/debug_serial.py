#!/usr/bin/env python3
"""
Debug tool: handshake with Pi, then send a test prompt and display raw output.

Usage:
    python debug_serial.py               # just listen (show raw bytes)
    python debug_serial.py --send        # handshake + send test prompt
    python debug_serial.py --raw         # raw bytes, no handshake
"""
import serial
import sys
import glob
import time
import struct

SOH = 0x04
STX = 0x02
ETX = 0x03
ACK = 0x06

def find_port():
    for pattern in ["/dev/cu.usbserial-*", "/dev/cu.SLAB_USB*"]:
        ports = glob.glob(pattern)
        if ports:
            return sorted(ports)[0]
    return None

args = [a for a in sys.argv[1:] if not a.startswith("--")]
flags = [a for a in sys.argv[1:] if a.startswith("--")]
port = args[0] if args else find_port()
SEND_TEST = "--send" in flags
RAW_MODE = "--raw" in flags

if not port:
    print("No serial port found")
    sys.exit(1)

print(f"Opening {port} @ 115200...")
ser = serial.Serial(port, 115200, timeout=1)
ser.reset_input_buffer()

def print_data(data, label=""):
    """Print hex + ASCII dump of data."""
    hex_str = " ".join(f"{b:02x}" for b in data)
    ascii_str = "".join(
        chr(b) if 32 <= b < 127 else
        ("\\n" if b == 10 else ("\\r" if b == 13 else f"[{b:02x}]"))
        for b in data
    )
    if label:
        print(f"--- {label} ---")
    print(f"HEX: {hex_str}")
    print(f"ASC: {ascii_str}")
    print()

def wait_for_ready():
    """Wait for READY protocol line from Pi, then send ACK."""
    print("Waiting for Pi READY...")
    buf = bytearray()
    in_proto = False
    while True:
        data = ser.read(256)
        if not data:
            continue
        for b in data:
            if b == SOH:
                in_proto = True
                buf = bytearray()
            elif b == ord("\n") and in_proto:
                line = buf.decode("utf-8", errors="replace")
                in_proto = False
                if line == "READY":
                    print("Got READY — sending ACK")
                    ser.write(bytes([ACK]))
                    time.sleep(0.2)  # let Pi process ACK
                    return
                else:
                    print(f"  Pi protocol: {line}")
            elif in_proto:
                buf.append(b)
            else:
                pass  # debug output, ignore

def send_test_prompt():
    """Send a test prompt and display everything that comes back."""
    prompt = "I: hello\n"
    frame = bytearray()
    frame.append(STX)
    frame.extend(prompt.encode("utf-8"))
    frame.extend(b"TIMEOUT:5000\n")
    frame.append(ETX)

    print(f"Sending prompt: {frame!r}")
    ser.write(frame)

    # Listen for response
    print("Listening for response (30s timeout)...")
    start = time.time()
    while time.time() - start < 30:
        data = ser.read(256)
        if data:
            print_data(data, f"t={time.time()-start:.1f}s")

try:
    if RAW_MODE:
        print("Raw mode — no handshake, just dumping bytes...\n")
        while True:
            data = ser.read(256)
            if data:
                print_data(data)
    elif SEND_TEST:
        wait_for_ready()
        print()
        send_test_prompt()
    else:
        print("Listening (Ctrl-C to quit)...\n")
        while True:
            data = ser.read(256)
            if data:
                print_data(data)
except KeyboardInterrupt:
    print("\nDone.")
finally:
    ser.close()
