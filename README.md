# Ghost in the Machine

[![Demo Video](https://img.youtube.com/vi/KBqjMb9z13o/maxresdefault.jpg)](https://youtu.be/KBqjMb9z13o)

I put myself in a Raspberry Pi.

This is a 15M parameter LLM trained from scratch on my text messages with my girlfriend, running bare-metal (no operating system) on a Raspberry Pi Zero (512MB RAM, $15). You plug the Pi into your computer and chat with it through a messaging webapp. It'll message you if you leave it alone for too long, and it won't automatically respond to every message you send.

This is the public version of this repo (without my messages). If you want to see the full commit history or private data, come ask me.

## Why

It's Valentine's weekend and I can't spend it with my girlfriend, so I made a clone of myself based on our 100k+ messages. I thought an LLM would be a good starting point, and because she doesn't have much coding experience or access to hardware, I wanted this to run on the cheapest computer around.

On a broader scale: how small can we make language models that still learn domain-specific knowledge, and what's the bare minimum hardware to serve them? These are important questions for distributing LLMs at a fraction of their current cost.

## How It Works

Three pieces:
1. **Training**: Fine-tune a 15M param Llama 2-style model on my text messages using PyTorch.
2. **Inference**: Run the quantized (int8) model in C on a bare-metal Raspberry Pi Zero.
3. **Webapp**: A messaging UI connected to the Pi over serial (UART) via a Python bridge.

```
Chat History → Tokenize → Fine-tune → Export → Bake into Pi kernel
                                                       ↓
                                              Raspberry Pi (bare metal)
                                                       ↓ (serial @ 115200 baud)
                                                  Host PC (server.py)
                                                       ↓ (WebSocket)
                                                  Browser (index.html)
```

## Setup

### Prerequisites
- `arm-none-eabi-gcc` (for cross-compiling to Pi)
- A Raspberry Pi Zero (or similar) with a serial connection to your host

### 1. Training the Model

Everything lives in `llama2.c/`.

**Install dependencies:**
```bash
cd llama2.c
pip install -r requirements.txt
```

**Download a base model** (pre-trained on TinyStories):
```bash
# 15M params (fits in Pi RAM after quantization)
wget https://huggingface.co/karpathy/tinyllamas/resolve/main/stories15M.pt

# 42M params (if you're not targeting the Pi)
wget https://huggingface.co/karpathy/tinyllamas/resolve/main/stories42M.pt -P finetunes
```

**Prepare your data:**

Put your cleaned chat history in `private/chat_history_train.txt` and `private/chat_history_val.txt`. Format is lines prefixed with `I:` (you) and `M:` (the other person):
```
I: hey what's up
M: not much just got home
I: nice want to get food
M: yes please I'm starving
```

Then tokenize and preprocess:
```bash
python chat_history.py pretokenize
python preprocess_chat.py --input_dir ../private --output_dir ../private
```

This creates binary token files and turn-offset indices so the training loop can sample from complete conversation turns (not random positions in the middle of a sentence).

**Fine-tune:**

On CUDA:
```bash
python finetune_msg.py \
  --device=cuda \
  --compile=True \
  --batch_size=64 \
  --warmup_iters=500 \
  --learning_rate=1e-5 \
  --dropout=0.1 \
  --wandb_log=True \
  --wandb_project=pi-ris \
  --wandb_run_name=run_nn
```

On Apple Silicon (MPS):
```bash
python finetune_msg.py \
  --device=mps \
  --compile=False \
  --batch_size=32 \
  --wandb_log=True \
  --wandb_project=pi-ris \
  --wandb_run_name=run_nn
```

The model fine-tunes from a TinyStories checkpoint (`stories15M.pt` by default). To use 42M params instead, pass `--ckpt_path=stories42M.pt`. Output checkpoint lands in `finetunes/ckpt.pt`.

**Test your model:**
```bash
python sample.py \
  --checkpoint=finetunes/ckpt.pt \
  --device=cuda \
  --compile=True \
  --num_samples=5 \
  --max_new_tokens=100 \
  --start="I: Hello\nM:"
```

### 2. Deploying to the Pi

The bare-metal inference code lives in `pi-baremetal/inference-server/`.

**Export your model to binary:**
```bash
python export.py finetunes/ckpt.pt
```
This creates a quantized `.bin` file that the C inference engine can read.

**Build the kernel image:**
```bash
cd pi-baremetal/inference-server
make MODEL=path/to/your/quantized_model.bin
```

This does a few things:
- Compiles `main.c` (protocol handler) and `llama2.c` (inference engine) with libpi
- Uses `objcopy` to bake the model weights and tokenizer directly into the kernel binary
- Outputs `kernel.img` (~17-18 MB)

**Flash to SD card:**

Copy `kernel.img` to your Pi's SD card (alongside `bootcode.bin` and `start.elf` from the Raspberry Pi firmware). The bootloader loads the kernel into RAM at `0x8000` and starts inference.

One forward pass takes ~2.2s on the Pi Zero. Not fast, but it works.

### 3. Running the Webapp

The webapp lives in `webapp/`.

**Install dependencies:**
```bash
cd webapp
pip install -r requirements.txt
```

**Start the bridge:**
```bash
python server.py /dev/ttyUSB0 115200
```
(Replace `/dev/ttyUSB0` with your actual serial port — on Mac it's usually something like `/dev/tty.usbserial-*`.)

This starts:
- An HTTP server on `http://localhost:8080` (serves the chat UI)
- A WebSocket server on `ws://localhost:8765` (real-time comms)
- A serial bridge to the Pi

**Open the chat:**

Go to `http://localhost:8080` in your browser. The UI mimics iMessage. It'll show a loading state until the Pi handshake completes, then you can start chatting.

**Debug tool:**

If something's weird with the serial connection:
```bash
python debug_serial.py              # listen mode (hex + ASCII dump)
python debug_serial.py --send       # handshake + send a test prompt
python debug_serial.py --raw        # raw mode (no protocol)
```

## Architecture Details

### Model
- Llama 2-style transformer (from [llama2.c](https://github.com/karpathy/llama2.c))
- 288-dim, 6 layers, 6 heads
- Vocab size: 32,000 (Llama 2 SentencePiece tokenizer)
- Sequence length: 256 tokens
- Quantized to int8 for inference (4x smaller weights)
- Pre-trained on TinyStories, fine-tuned on ~20k chat messages

### Bare-Metal Inference
- Pure C, no OS, no stdlib (custom libpi)
- Hardware floating point (VFP) enabled
- KV cache kept in float32 for quality, weights in int8 for size
- Model baked directly into kernel binary via `objcopy`
- Custom UART protocol with handshaking, cancel support, and retry logic

### Communication Protocol
The Pi and host communicate over serial with a custom protocol:
1. **Handshake**: Pi sends `READY` every 500ms until host responds with `ACK`
2. **Prompt**: Host sends `STX + prompt + ETX`, Pi responds with `OK`
3. **Generation**: Pi sends `TYPING` → `MSG:<text>` → `IDLE` as it generates
4. **Cancel**: Host sends `CAN` byte, Pi acknowledges and stops

The protocol handles race conditions (messages sent mid-generation) and retries (if the model doesn't produce a valid `M:` response).

## Things I Learned

- There's way more skill expression in how you prepare data than in hyperparameter tuning. My model went from incoherent garbage to passable English mostly by cleaning the training data better.
- The KV cache is everything. You can checkpoint it between turns instead of recomputing from scratch, which saves a ton of inference time.
- Watching randomly initialized weights learn my actual speech patterns was wild. I still can't fully explain how the simplest possible training pipeline produced something that sounds like me.
- Dropout actually works. My val loss curve literally inverted itself when I added some.

## What's Next

- **More data**: I couldn't download my last year of Messenger data in time. That's a significant chunk.
- **Faster inference**: Haven't really optimized the C code beyond quantization. The Pi Zero has a GPU of sorts that might be faster at matmuls. Also want to look at [llama.cpp](https://github.com/ggerganov/llama.cpp) for tricks.
- **Better training**: Curious about pre-training on general conversation data first, then fine-tuning with frozen layers. Also want to understand the failure modes of my synthetic data generation.

## Project Structure
```
pi-ris-public/
├── llama2.c/                    # Training & model code
│   ├── finetune_msg.py          # Fine-tuning script
│   ├── chat_history.py          # Tokenization pipeline
│   ├── preprocess_chat.py       # Turn-boundary preprocessing
│   ├── sample.py                # Inference sampling (PyTorch)
│   ├── model.py                 # Transformer architecture
│   ├── export.py                # Model export to binary
│   ├── train.py                 # Base training script
│   ├── run.c                    # Float32 C inference
│   └── runq.c                   # Int8 quantized C inference
│
├── pi-baremetal/                # Raspberry Pi bare-metal code
│   ├── libpi/                   # Core Pi library (GPIO, UART, memory)
│   ├── bootloader/              # Bootstrap loader
│   └── inference-server/        # LLM inference on bare metal
│       ├── main.c               # Protocol handler + inference loop
│       ├── llama2.c             # Quantized inference engine (C)
│       └── Makefile             # Builds kernel.img
│
├── webapp/                      # Web UI + serial bridge
│   ├── server.py                # Async serial-WebSocket-HTTP bridge
│   ├── index.html               # iMessage-style chat UI
│   └── debug_serial.py          # Serial debugging tool
│
└── private/                     # Your chat data goes here (gitignored)
    ├── chat_history_train.txt
    └── chat_history_val.txt
```

## Acknowledgements
- [llama2.c](https://github.com/karpathy/llama2.c) — training and inference code. This project would not be possible without it.
- Eric Chen — guidance during the model training process.
- Iris Nguyen — the idea + the data :)
