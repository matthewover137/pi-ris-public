#!/usr/bin/env python3
"""
Bridge between the iMessage web UI and the Pi inference server.

Protocol:
  1. Pi sends READY every 500ms. Host sees READY → sends ACK. Pi stops.
  2. Host sends STX + prompt + ETX. Pi reads it, sends OK.
  3. Pi generates, sends TYPING/MSG/IDLE/NOREPLY/RETRY.
  4. Host sends CAN to cancel. Pi sends CANCEL_ACK.

Usage:
    python server.py [serial_port] [baud_rate]
"""

import asyncio
import json
import glob
import sys
import os
import threading
from http.server import HTTPServer, SimpleHTTPRequestHandler
from functools import partial

import serial
import websockets

# Protocol constants (must match llama2.h)
STX = 0x02
ETX = 0x03
SOH = 0x04
ACK = 0x06
CAN = 0x18

HTTP_PORT = 8080
WS_PORT = 8765
BAUD_RATE = 115200


def find_serial_port():
    patterns = [
        "/dev/cu.usbserial-*",
        "/dev/cu.SLAB_USB*",
        "/dev/ttyUSB*",
        "/dev/ttyACM*",
    ]
    for pattern in patterns:
        ports = glob.glob(pattern)
        if ports:
            return sorted(ports)[0]
    return None


class Bridge:
    def __init__(self, port, baud):
        self.port = port
        self.baud = baud
        self.ser = None
        self.ws = None
        self.running = True
        self.pi_ready = False
        self.pi_busy = False  # True when Pi is generating/retrying
        # Events for synchronization
        self.ok_event = asyncio.Event()
        self.cancel_ack_event = asyncio.Event()

    def connect_serial(self):
        self.ser = serial.Serial(self.port, self.baud, timeout=0.1)
        self.ser.reset_input_buffer()
        print(f"  Serial: {self.port} @ {self.baud}")

    # --- Debug logging ---

    CTRL_NAMES = {0x02: "STX", 0x03: "ETX", 0x04: "SOH", 0x06: "ACK", 0x18: "CAN"}

    def _fmt_bytes(self, data):
        """Format bytes as readable string: control chars as names, rest as ASCII."""
        parts = []
        for b in data:
            if b in self.CTRL_NAMES:
                parts.append(f"[{self.CTRL_NAMES[b]}]")
            elif b == 0x0a:
                parts.append("\\n")
            elif b == 0x0d:
                parts.append("\\r")
            elif 32 <= b < 127:
                parts.append(chr(b))
            else:
                parts.append(f"[{b:02x}]")
        return "".join(parts)

    def _log_rx(self, data):
        if data:
            print(f"  RX: {self._fmt_bytes(data)}")

    def _log_tx(self, data):
        if data:
            print(f"  TX: {self._fmt_bytes(data)}")

    # --- Protocol line parser (shared by handshake + main loop) ---

    def _parse_bytes(self, data, line_buf, in_protocol):
        """Parse raw serial bytes into protocol lines. Returns (lines, line_buf, in_protocol)."""
        self._log_rx(data)
        lines = []
        for byte in data:
            if byte == SOH:
                in_protocol = True
                line_buf = bytearray()
            elif byte == ord("\n") and in_protocol:
                lines.append(line_buf.decode("utf-8", errors="replace"))
                in_protocol = False
            elif in_protocol:
                line_buf.append(byte)
        return lines, line_buf, in_protocol

    # --- Phase 1: Handshake ---

    async def handshake(self):
        """Wait for READY from Pi, send ACK."""
        loop = asyncio.get_event_loop()
        print("  Waiting for Pi READY...")
        line_buf = bytearray()
        in_proto = False

        while True:
            try:
                data = await loop.run_in_executor(None, self.ser.read, 256)
            except Exception as e:
                print(f"  Serial error: {e}")
                await asyncio.sleep(1)
                continue
            if not data:
                continue

            lines, line_buf, in_proto = self._parse_bytes(data, line_buf, in_proto)
            for line in lines:
                if line == "READY":
                    self._log_tx(bytes([ACK]))
                    self.ser.write(bytes([ACK]))
                    self.pi_ready = True
                    print("  Got READY → sent ACK. Handshake complete!")
                    return

    # --- Serial reader (main loop after handshake) ---

    async def serial_reader(self):
        loop = asyncio.get_event_loop()
        line_buf = bytearray()
        in_proto = False

        while self.running:
            try:
                data = await loop.run_in_executor(None, self.ser.read, 256)
            except serial.SerialException as e:
                print(f"  Serial error: {e}")
                await asyncio.sleep(1)
                continue
            except Exception as e:
                print(f"  Read error: {e}")
                await asyncio.sleep(0.5)
                continue

            if not data:
                continue

            lines, line_buf, in_proto = self._parse_bytes(data, line_buf, in_proto)
            for line in lines:
                await self.handle_protocol_line(line)

    async def handle_protocol_line(self, line):
        print(f"  Pi → {line}")

        # Internal handshake signals
        if line == "OK":
            self.pi_busy = True
            self.ok_event.set()
            return
        if line == "CANCEL_ACK":
            self.pi_busy = False
            self.cancel_ack_event.set()
            return
        if line == "READY":
            # Pi rebooted — re-ACK
            self._log_tx(bytes([ACK]))
            self.ser.write(bytes([ACK]))
            print("  Pi rebooted — re-sent ACK")
            return
        if line.startswith("DBG:"):
            # Debug output from generation — log only, don't forward
            print(f"  Pi DBG: {line[4:]}")
            return

        # Forward to browser
        if not self.ws:
            return
        try:
            if line == "TYPING":
                await self.ws.send(json.dumps({"type": "typing"}))
            elif line.startswith("MSG:"):
                await self.ws.send(json.dumps({"type": "message", "text": line[4:]}))
            elif line == "IDLE":
                self.pi_busy = False
                await self.ws.send(json.dumps({"type": "idle"}))
            elif line == "NOREPLY":
                await self.ws.send(json.dumps({"type": "noreply"}))
            elif line == "RETRY":
                await self.ws.send(json.dumps({"type": "retry"}))
        except websockets.exceptions.ConnectionClosed:
            self.ws = None

    # --- Send prompt with retry ---

    def _send_frame(self, history, timeout_ms):
        """Send STX + prompt + TIMEOUT + ETX."""
        frame = bytearray()
        frame.append(STX)
        frame.extend(history.encode("utf-8"))
        frame.extend(f"TIMEOUT:{timeout_ms}\n".encode("utf-8"))
        frame.append(ETX)
        self._log_tx(frame)
        self.ser.write(frame)

    async def send_prompt(self, history, timeout_ms):
        """Send prompt and wait for OK. No retry — retrying a frame is
        dangerous because leftover bytes from the first frame would
        corrupt the Pi's state. If OK doesn't come, something is
        seriously wrong (Pi crashed, cable unplugged)."""
        self.ok_event.clear()
        self._send_frame(history, timeout_ms)
        print(f"  Host → prompt ({len(history)} chars)")
        try:
            await asyncio.wait_for(self.ok_event.wait(), timeout=5.0)
            print("  Pi → OK (prompt acknowledged)")
            return True
        except asyncio.TimeoutError:
            print("  ERROR: No OK from Pi after 5s — Pi may have crashed")
            return False

    async def send_cancel(self):
        """Send CAN and wait for CANCEL_ACK (with timeout)."""
        self.cancel_ack_event.clear()
        self._log_tx(bytes([CAN]))
        self.ser.write(bytes([CAN]))
        try:
            await asyncio.wait_for(self.cancel_ack_event.wait(), timeout=3.0)
            print("  Pi → CANCEL_ACK")
            return True
        except asyncio.TimeoutError:
            print("  No CANCEL_ACK after 3s")
            self.pi_busy = False  # assume it worked
            return False

    # --- WebSocket handler ---

    async def ws_handler(self, websocket):
        print("  Browser connected")
        self.ws = websocket

        if self.pi_ready:
            await websocket.send(json.dumps({"type": "ready"}))

        try:
            async for message in websocket:
                data = json.loads(message)
                if data["type"] == "send":
                    # Cancel if busy
                    if self.pi_busy:
                        await self.send_cancel()
                    # Send new prompt
                    await self.send_prompt(
                        data["history"],
                        data.get("timeout", 5000),
                    )
        except websockets.exceptions.ConnectionClosed:
            pass
        finally:
            print("  Browser disconnected")
            self.ws = None


def start_http_server(directory, port):
    handler = partial(SimpleHTTPRequestHandler, directory=directory)
    httpd = HTTPServer(("localhost", port), handler)
    httpd.serve_forever()


async def main():
    port = sys.argv[1] if len(sys.argv) > 1 else find_serial_port()
    baud = int(sys.argv[2]) if len(sys.argv) > 2 else BAUD_RATE

    if not port:
        print("No serial port found.")
        print(f"  python {sys.argv[0]} /dev/cu.usbserial-110")
        sys.exit(1)

    bridge = Bridge(port, baud)
    bridge.connect_serial()

    # HTTP server
    webapp_dir = os.path.dirname(os.path.abspath(__file__))
    http_thread = threading.Thread(
        target=start_http_server,
        args=(webapp_dir, HTTP_PORT),
        daemon=True,
    )
    http_thread.start()
    print(f"  Web UI: http://localhost:{HTTP_PORT}")
    print(f"  WebSocket: ws://localhost:{WS_PORT}")

    # Handshake
    await bridge.handshake()

    if bridge.ws:
        await bridge.ws.send(json.dumps({"type": "ready"}))

    # Run serial reader + websocket server
    async with websockets.serve(bridge.ws_handler, "localhost", WS_PORT):
        await bridge.serial_reader()


if __name__ == "__main__":
    print("=== Pi Chat Bridge ===")
    asyncio.run(main())
