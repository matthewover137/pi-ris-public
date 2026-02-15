# This is the public version of this, without my messages. If you wanna ask about my commit history or see my whole codebase come ask me please thanks!
steps:
0. wget https://huggingface.co/karpathy/tinyllamas/resolve/main/stories15M.pt (base model)
1. run the data cleaner notebook in private
2. run python chat_history.py pretokenize to tokenize the data
3. run python finetune_msg.py --device=mps --compile=False --batch_size=32 --wandb_log=True --wandb_project=pi-ris --wandb_run_name=run_nn
   or python finetune_msg.py --device=cuda --compile=True --batch_size=64 --wandb_log=True --wandb_project=pi-ris --wandb_run_name=run_nn

test: 
4. run python sample.py --checkpoint=finetunes/ckpt.pt --device=cuda --compile=True --num_samples=5 --max_new_tokens=100 --start="I: Hello\nM:" 


python finetune_msg.py --device=cuda --compile=True --batch_size=64  --warmup_iters=500 --learning_rate=1e-5 --dropout=0.1  --wandb_log=True --wandb_project=pi-ris --wandb_run_name=run_nn
python finetune_msg.py --device=cuda --compile=True --batch_size=64  --warmup_iters=500 --learning_rate=1e-5 --dropout=0.1  --wandb_log=True --wandb_project=pi-ris --wandb_run_name=run_nn
--dropout=0.1 


TODO: clean data better:
I: Hello\nM: Iris missed your video call.
I: Iris missed your video call.
I: Iris made an update.
I: Iris sent an attachment.
Iris set the quick reaction to
--ckpt_path=stories42M.pt
python finetune_msg.py --ckpt_path=stories42M.pt --device=cuda --compile=True --batch_size=64  --warmup_iters=500 --learning_rate=1e-5 --dropout=0.1  --wandb_log=True --wandb_project=pi-ris --wandb_run_name=run_nn
wget https://huggingface.co/karpathy/tinyllamas/resolve/main/stories42M.pt -P finetunes


can try: 
more data
freezing layers
blah blah
hyperparamters
blah blah