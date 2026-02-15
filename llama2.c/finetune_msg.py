#ADDED CODE - MLi
import math
import os
import time
from contextlib import nullcontext
from datetime import datetime
import numpy as np # Added for memmap

import torch
from model import Transformer, ModelArgs
from torch.distributed import destroy_process_group, init_process_group
from torch.nn.parallel import DistributedDataParallel as DDP
from export import model_export

# I/O
out_dir = "finetunes"
ckpt_path = "stories42M.pt"
eval_interval = 500
log_interval = 10
eval_iters = 100
eval_only = False  # if True, script exits right after the first eval
always_save_checkpoint = False  # if True, always save a checkpoint after each eval
init_from = "resume"  # 'scratch' or 'resume'
# wandb logging
wandb_log = False
wandb_project = "pi-ris"
wandb_run_name = "run" + datetime.now().strftime("%Y_%m_%d_%H_%M_%S")

# Data Loading Config
batch_size = 64  # Adjusted for typical GPU memory
max_seq_len = 256
vocab_size = 32000 # Llama 2 tokenizer size

# Model Config
dim = 288
n_layers = 6
n_heads = 6
n_kv_heads = 6
multiple_of = 32
dropout = 0.0

# AdamW Optimizer
gradient_accumulation_steps = 1 
learning_rate = 5e-4
max_iters = 5000  # Reduced: Chat history is smaller than TinyStories
weight_decay = 1e-1
beta1 = 0.9
beta2 = 0.95
grad_clip = 1.0

# Learning Rate Decay
decay_lr = True
warmup_iters = 100

# System
device = "cuda"
dtype = "bfloat16"
compile = True

config_keys = [
    k
    for k, v in globals().items()
    if not k.startswith("_") and isinstance(v, (int, float, bool, str))
]
exec(open("configurator.py").read())
config = {k: globals()[k] for k in config_keys}

# Fixed Hyperparams
lr_decay_iters = max_iters
min_lr = 0.0

# DDP Setup
ddp = int(os.environ.get("RANK", -1)) != -1
if ddp:
    init_process_group(backend="nccl")
    ddp_rank = int(os.environ["RANK"])
    ddp_local_rank = int(os.environ["LOCAL_RANK"])
    ddp_world_size = int(os.environ["WORLD_SIZE"])
    device = f"cuda:{ddp_local_rank}"
    torch.cuda.set_device(device)
    master_process = ddp_rank == 0
    seed_offset = ddp_rank
else:
    master_process = True
    seed_offset = 0
    ddp_world_size = 1
    ddp_local_rank = 0

if master_process:
    os.makedirs(out_dir, exist_ok=True)
torch.manual_seed(1337 + seed_offset)
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
device_type = "cuda" if "cuda" in device else "cpu"
ptdtype = {"float32": torch.float32, "bfloat16": torch.bfloat16, "float16": torch.float16}[dtype]
ctx = (
    nullcontext()
    if device_type == "cpu"
    else torch.amp.autocast(device_type=device_type, dtype=ptdtype)
)

# -----------------------------------------------------------------------------
# CUSTOM DATA LOADER — Samples from 'I:'/'M:' turn boundaries
# -----------------------------------------------------------------------------
# Requires preprocessed files from preprocess_chat.py:
#   chat_history_train.bin, chat_history_val.bin
#   turn_offsets_train.npy, turn_offsets_val.npy

data_dir = "./../private"

def get_data_path(filename):
    return os.path.join(data_dir, filename)

# Load tokenized data
train_data = np.memmap(get_data_path("chat_history_train.bin"), dtype=np.uint16, mode='r')
val_data = np.memmap(get_data_path("chat_history_val.bin"), dtype=np.uint16, mode='r')

# Load turn offset indices (positions where 'I:'/'M:' turns start)
train_offsets = np.load(get_data_path("turn_offsets_train.npy"))
val_offsets = np.load(get_data_path("turn_offsets_val.npy"))

# Filter out offsets too close to the end of the data for a full sequence
train_offsets = train_offsets[train_offsets + max_seq_len < len(train_data)]
val_offsets = val_offsets[val_offsets + max_seq_len < len(val_data)]

print(f"Training: {len(train_offsets)} usable turn positions out of {len(train_data)} tokens")
print(f"Validation: {len(val_offsets)} usable turn positions out of {len(val_data)} tokens")

if len(train_offsets) < batch_size:
    raise ValueError(
        f"Only {len(train_offsets)} usable training positions but batch_size={batch_size}. "
        f"Reduce batch_size or add more data."
    )

def iter_batches(split):
    data = train_data if split == "train" else val_data
    offsets = train_offsets if split == "train" else val_offsets

    while True:
        # Randomly sample from turn boundary positions
        ix = offsets[torch.randint(len(offsets), (batch_size,))]

        # Build input (x) and target (y) tensors
        x = torch.stack([torch.from_numpy((data[i : i + max_seq_len]).astype(np.int64)) for i in ix])
        y = torch.stack([torch.from_numpy((data[i + 1 : i + 1 + max_seq_len]).astype(np.int64)) for i in ix])

        if device_type == "cuda":
            x, y = x.pin_memory().to(device, non_blocking=True), y.pin_memory().to(device, non_blocking=True)
        else:
            x, y = x.to(device), y.to(device)

        yield x, y
# -----------------------------------------------------------------------------

# # -----------------------------------------------------------------------------
# # CUSTOM DATA LOADER (Replaces Task.iter_batches)
# # -----------------------------------------------------------------------------
# data_dir = "./../private"

# def get_data_path(split):
#     return os.path.join(data_dir, f"chat_history_{split}.bin")

# # Check if files exist
# if not os.path.exists(get_data_path("train")):
#     raise FileNotFoundError(f"Could not find training data at {get_data_path('train')}")

# # Memory map the data (avoids loading distinct copies into RAM for every worker)
# train_data = np.memmap(get_data_path("train"), dtype=np.uint16, mode='r')
# val_data = np.memmap(get_data_path("val"), dtype=np.uint16, mode='r')

# def iter_batches(split):
#     data = train_data if split == "train" else val_data
#     # If data is too small for batch size, this will error. 
#     # Ensure your chat history is at least (batch_size * max_seq_len) tokens.
#     while True:
#         # Randomly sample chunks of data
#         ix = torch.randint(len(data) - max_seq_len, (batch_size,))
        
#         # Stack them into tensors
#         x = torch.stack([torch.from_numpy((data[i : i + max_seq_len]).astype(np.int64)) for i in ix])
#         y = torch.stack([torch.from_numpy((data[i + 1 : i + 1 + max_seq_len]).astype(np.int64)) for i in ix])
        
#         # Move to device
#         if device_type == "cuda":
#             # pin_memory().to(...) is faster for GPU
#             x, y = x.pin_memory().to(device, non_blocking=True), y.pin_memory().to(device, non_blocking=True)
#         else:
#             x, y = x.to(device), y.to(device)
        
#         yield x, y

# # -----------------------------------------------------------------------------

iter_num = 0
best_val_loss = 1e9

model_args = dict(
    dim=dim,
    n_layers=n_layers,
    n_heads=n_heads,
    n_kv_heads=n_kv_heads,
    vocab_size=vocab_size,
    multiple_of=multiple_of,
    max_seq_len=max_seq_len,
    dropout=dropout,
)

if init_from == "scratch":
    print("Initializing a new model from scratch BAD BAD BAD")
    gptconf = ModelArgs(**model_args)
    model = Transformer(gptconf)
elif init_from == "resume":
    print(f"Resuming training from {out_dir}")
    ckpt_path = os.path.join(out_dir, ckpt_path)
    checkpoint = torch.load(ckpt_path, map_location=device)
    checkpoint_model_args = checkpoint["model_args"]
    for k in ["dim", "n_layers", "n_heads", "n_kv_heads", "vocab_size", "multiple_of", "max_seq_len"]:
        model_args[k] = checkpoint_model_args[k]
    gptconf = ModelArgs(**model_args)
    model = Transformer(gptconf)
    state_dict = checkpoint["model"]
    unwanted_prefix = "_orig_mod."
    for k, v in list(state_dict.items()):
        if k.startswith(unwanted_prefix):
            state_dict[k[len(unwanted_prefix) :]] = state_dict.pop(k)
    
    model.load_state_dict(state_dict)

    print("FINE-TUNING DETECTED: Resetting iteration count and optimizer.")
    iter_num = 0             # Reset step count to 0 so training starts
    best_val_loss = 1e9      # Reset validation tracking

model.to(device)

scaler = torch.amp.GradScaler(enabled=(dtype == "float16"))
optimizer = model.configure_optimizers(weight_decay, learning_rate, (beta1, beta2), device_type)

# dont load the optimizer when finetuning
# if init_from == "resume" and "optimizer" in checkpoint:
#     optimizer.load_state_dict(checkpoint["optimizer"])

if compile:
    print("compiling the model... (takes a ~minute)")
    unoptimized_model = model
    model = torch.compile(model)

if ddp:
    model = DDP(model, device_ids=[ddp_local_rank])

@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in ["train", "val"]:
        batch_iter = iter_batches(split=split)
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = next(batch_iter)
            with ctx:
                logits = model(X, Y)
                loss = raw_model.last_loss
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out

def get_lr(it):
    if it < warmup_iters:
        return learning_rate * it / warmup_iters
    if it > lr_decay_iters:
        return min_lr
    decay_ratio = (it - warmup_iters) / (lr_decay_iters - warmup_iters)
    assert 0 <= decay_ratio <= 1
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return min_lr + coeff * (learning_rate - min_lr)

if wandb_log and master_process:
    import wandb
    wandb.init(project=wandb_project, name=wandb_run_name, config=config)

train_batch_iter = iter_batches(split="train")
X, Y = next(train_batch_iter)
t0 = time.time()
local_iter_num = 0
raw_model = model.module if ddp else model
running_mfu = -1.0

while True:
    lr = get_lr(iter_num) if decay_lr else learning_rate
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr

    if iter_num % eval_interval == 0 and master_process:
        losses = estimate_loss()
        print(f"step {iter_num}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")
        if wandb_log:
            wandb.log({
                "iter": iter_num,
                "loss/train": losses["train"],
                "loss/val": losses["val"],
                "lr": lr,
            }, step=iter_num)
        
        if losses["val"] < best_val_loss or always_save_checkpoint:
            best_val_loss = losses["val"]
            if iter_num > 0:
                checkpoint = {
                    "model": raw_model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "model_args": model_args,
                    "iter_num": iter_num,
                    "best_val_loss": best_val_loss,
                    "config": config,
                }
                print(f"saving checkpoint to {out_dir}")
                torch.save(checkpoint, os.path.join(out_dir, "ckpt.pt"))
                model_export(raw_model, os.path.join(out_dir, "model.bin"), version=1)

    if iter_num == 0 and eval_only:
        break

    for micro_step in range(gradient_accumulation_steps):
        if ddp:
            model.require_backward_grad_sync = micro_step == gradient_accumulation_steps - 1
        with ctx:
            logits = model(X, Y)
            loss = raw_model.last_loss
            loss = loss / gradient_accumulation_steps
        X, Y = next(train_batch_iter)
        scaler.scale(loss).backward()

    if grad_clip != 0.0:
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
    
    scaler.step(optimizer)
    scaler.update()
    optimizer.zero_grad(set_to_none=True)

    t1 = time.time()
    dt = t1 - t0
    t0 = t1
    if iter_num % log_interval == 0 and master_process:
        lossf = loss.item() * gradient_accumulation_steps
        if local_iter_num >= 5:
            mfu = raw_model.estimate_mfu(batch_size * gradient_accumulation_steps, dt)
            running_mfu = mfu if running_mfu == -1.0 else 0.9 * running_mfu + 0.1 * mfu
        print(f"{iter_num} | loss {lossf:.4f} | lr {lr:e} | {dt*1000:.2f}ms | mfu {running_mfu*100:.2f}%")

    iter_num += 1
    local_iter_num += 1

    if iter_num > max_iters:
        break

if ddp:
    destroy_process_group()