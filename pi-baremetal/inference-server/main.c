// Pi bare-metal inference server — chat protocol handler
//
// Protocol (inspired by bootloader handshake):
//
//   PHASE 1 — READY handshake
//     Pi:   sends \x04READY\n every 500ms
//     Host: discards bytes until it sees READY, then sends ACK (0x06)
//     Pi:   receives ACK, stops sending READY → ready for prompts
//
//   PHASE 2 — Prompt exchange
//     Host: sends STX (0x02)
//     Host: sends prompt bytes (text)
//     Host: sends ETX (0x03)
//     Pi:   reads bytes between STX and ETX
//     Pi:   sends \x04OK\n (prompt received)
//     Host: waits for OK before proceeding
//
//   PHASE 3 — Generation
//     Pi:   generates tokens, sends protocol lines:
//           \x04TYPING\n     (show typing indicator)
//           \x04MSG:<text>\n (complete message)
//           \x04IDLE\n       (done, go back to phase 2)
//           \x04NOREPLY\n    (no M: produced, entering retry wait)
//           \x04RETRY\n      (retrying generation)
//
//   PHASE 4 — Cancel (during generation or retry wait)
//     Host: sends CAN (0x18)
//     Pi:   detects CAN, sends \x04CANCEL_ACK\n
//     Host: waits for CANCEL_ACK, then can send new prompt (phase 2)
//
#include "rpi.h"
#include "memmap.h"
#include "llama2.h"

// Baked-in data symbols (created by objcopy)
extern const uint8_t _binary_msg_model_bin_start[];
extern const uint8_t _binary_msg_model_bin_end[];
extern const uint8_t _binary_tokenizer_bin_start[];
extern const uint8_t _binary_tokenizer_bin_end[];

// Enable the hardware VFP (floating point unit)
static void enable_vfp(void) {
    unsigned val;
    asm volatile("mrc p15, 0, %0, c1, c0, 2" : "=r"(val));
    val |= (0xF << 20);
    asm volatile("mcr p15, 0, %0, c1, c0, 2" :: "r"(val));
    asm volatile("mcr p15, 0, %0, c7, c5, 4" :: "r"(0));
    asm volatile("fmxr fpexc, %0" :: "r"(1 << 30));
}

// Send a protocol line: \x04<text>\n
static void proto_send(const char *line) {
    uart_put8(SOH);
    for (const char *p = line; *p; p++)
        uart_put8(*p);
    uart_put8('\n');
}

// Drain any garbage bytes from the UART RX FIFO
static void uart_drain(void) {
    while (uart_has_data())
        uart_get8();
}

// Parse "TIMEOUT:<ms>" from the end of the prompt buffer.
// Strips it and returns timeout in ms. Returns default if not found.
static int parse_timeout(char *buf, int len) {
    int timeout_ms = 5000;
    for (int i = len - 1; i >= 8; i--) {
        if (buf[i-8] == 'T' && buf[i-7] == 'I' && buf[i-6] == 'M' &&
            buf[i-5] == 'E' && buf[i-4] == 'O' && buf[i-3] == 'U' &&
            buf[i-2] == 'T' && buf[i-1] == ':') {
            timeout_ms = 0;
            for (int j = i; j < len && buf[j] >= '0' && buf[j] <= '9'; j++)
                timeout_ms = timeout_ms * 10 + (buf[j] - '0');
            int strip_start = i - 8;
            if (strip_start > 0 && buf[strip_start - 1] == '\n')
                strip_start--;
            buf[strip_start] = '\0';
            break;
        }
    }
    return timeout_ms;
}

// Wait for timeout_ms, checking for cancel (CAN on UART).
// Returns 1 if cancelled (sends CANCEL_ACK), 0 if timeout expired.
static int wait_or_cancel(int timeout_ms) {
    for (int elapsed = 0; elapsed < timeout_ms; elapsed += 10) {
        delay_ms(10);
        if (uart_has_data()) {
            int c = uart_get8();
            if (c == CAN) {
                proto_send("CANCEL_ACK");
                return 1;
            }
        }
    }
    return 0;
}

void notmain(void) {
    enable_vfp();
    caches_enable();
    kmalloc_init_set_start(__heap_start__, 64 * 1024 * 1024);

    // --- Boot: load model ---
    printk("=== Pi Inference Server ===\n");

    Transformer transformer;
    printk("Loading model... ");
    build_transformer(&transformer, _binary_msg_model_bin_start);
    Config *c = &transformer.config;
    printk("done.\n");
    unsigned file_bytes = _binary_msg_model_bin_end - _binary_msg_model_bin_start;
    printk("  %dMB quantized, dim=%d, %d layers, %d heads, vocab=%d, seq_len=%d\n",
           file_bytes / (1024 * 1024),
           c->dim, c->n_layers, c->n_heads, c->vocab_size, c->seq_len);

    Tokenizer tokenizer;
    printk("Loading tokenizer... ");
    build_tokenizer(&tokenizer, _binary_tokenizer_bin_start, transformer.config.vocab_size);
    printk("done.\n");

    Sampler sampler;
    build_sampler(&sampler, transformer.config.vocab_size, 1.0f, 0.9f, timer_get_usec());
    int steps = transformer.config.seq_len;

    // --- Phase 1: READY handshake ---
    // Send READY every 500ms until host responds with ACK.
    // This handles: host starts before Pi, host starts after Pi,
    // or host reconnects after a disconnect.
    uart_drain();
    printk("Waiting for host ACK...\n");
    while (1) {
        proto_send("READY");
        // Wait up to 500ms for ACK, checking every 10ms
        for (int i = 0; i < 50; i++) {
            delay_ms(10);
            if (uart_has_data()) {
                int ch = uart_get8();
                if (ch == ACK) goto host_connected;
            }
        }
    }

host_connected:
    uart_drain(); // discard any extra bytes
    printk("Host connected. Chat active.\n");

    // --- Main protocol loop ---
    char prompt_buf[4096];

    while (1) {
        // --- Phase 2: Wait for prompt ---
        // Spin until we get STX. Discard anything else.
        while (1) {
            int ch = uart_get8();
            if (ch == STX) break;
        }

        // Read prompt bytes until ETX
        int len = 0;
        while (1) {
            int ch = uart_get8();
            if (ch == ETX) break;
            if (len < (int)sizeof(prompt_buf) - 1)
                prompt_buf[len++] = ch;
        }
        prompt_buf[len] = '\0';

        // Acknowledge receipt
        proto_send("OK");

        // Parse and strip TIMEOUT from end of prompt
        int timeout_ms = parse_timeout(prompt_buf, len);

        // --- Phase 3: Generation with retry loop ---
        while (1) {
            int result = chat_generate(&transformer, &tokenizer, &sampler,
                                       prompt_buf, steps);

            if (result == CHAT_COMPLETE) {
                proto_send("IDLE");
                break; // done — wait for next prompt
            }

            if (result == CHAT_CANCELLED) {
                // CANCEL_ACK already sent inside chat_generate
                // Go back to waiting for new prompt (Phase 2)
                break;
            }

            // NOREPLY — model didn't produce M:
            proto_send("NOREPLY");

            // Wait timeout, checking for cancel
            if (wait_or_cancel(timeout_ms)) {
                // Cancelled during wait — CANCEL_ACK already sent
                break;
            }

            // Timeout expired, retry
            proto_send("RETRY");
        }
    }
}
