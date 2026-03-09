import os
import numpy as np
from CrossEntropyLoss import CrossEntropyLoss
from NanoLLM import GPT2
import tiktoken
import config
import torch
import matplotlib.pyplot as plt
from tqdm import tqdm
from torch.optim import lr_scheduler
from torch.utils.tensorboard import SummaryWriter

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Using device: {torch.cuda.get_device_name(device) if device=='cuda' else 'cpu'}")

tokenizer = tiktoken.get_encoding("gpt2")
vocab_size = tokenizer.n_vocab
print(f"Vocab size: {vocab_size}")


def prepare_bin(txt_path, bin_path):
    if os.path.exists(bin_path):
        print(f"{bin_path} already exists, skipping tokenization.")
        return
    print(f"Tokenizing {txt_path}...")
    with open(txt_path, 'r') as f:
        raw_text = f.read()
    tokens = tokenizer.encode(raw_text, allowed_special={"<|endoftext|>"})
    np.array(tokens, dtype=np.uint16).tofile(bin_path)
    print(f"Saved {len(tokens)} tokens to {bin_path}")


def get_batch(split, context_size, batch_size, device):
    train_data = np.memmap('train.bin', dtype=np.uint16, mode='r')
    val_data = np.memmap('val.bin', dtype=np.uint16, mode='r')
    data = train_data if split == 'train' else val_data
    indices = torch.randint(low=0, high=len(data) - context_size, size=(batch_size,))
    x = torch.stack([torch.from_numpy((data[i.item():i.item()+context_size]).astype(np.int64)) for i in indices])
    y = torch.stack([torch.from_numpy((data[i.item()+1:i.item()+context_size+1]).astype(np.int64)) for i in indices])
    if 'cuda' in device:
        x, y = x.pin_memory().to(device, non_blocking=True), y.pin_memory().to(device, non_blocking=True)
    return x, y


prepare_bin("train.txt", "train.bin")
prepare_bin("val.txt", "val.bin")

if __name__ == "__main__":
    model = GPT2(
        vocab_size=vocab_size,
        context_size=config.context_length,
        num_embeddings=config.num_embeddings,
        num_heads=config.num_heads,
        num_blocks=config.num_blocks,
    )
    model = model.to(device)
    if config.use_compile:
        model = torch.compile(model)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=1e-5
    )
    scheduler = lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=config.num_epochs,
        eta_min=1e-7
    )
    criterion = CrossEntropyLoss()

    train_losses = []
    val_losses = []
    mem_allocated = []   # current GPU memory per epoch (GB)
    mem_peak = []        # peak GPU memory per epoch (GB)
    gpu_utilization = [] # GPU utilization % per epoch
    best_val_loss = float('inf')
    writer = SummaryWriter('runs/training_logs')

    print("Starting training...")
    for epoch in tqdm(range(config.num_epochs), desc="Epochs"):
        model.train()
        epoch_loss = 0.0
        batch_pbar = tqdm(range(config.steps_per_epoch), desc=f"Epoch {epoch+1}", leave=False)

        for batch_idx in batch_pbar:
            inputs, targets = get_batch('train', config.context_length, config.batch_size, device)
            optimizer.zero_grad()

            with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
                logits = model(inputs)
                loss = criterion(logits, targets)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            # Log gradients and weights on first batch of each epoch
            if batch_idx == 0:
                global_step = epoch * config.steps_per_epoch + batch_idx
                for name, param in model.named_parameters():
                    if param.grad is not None:
                        writer.add_histogram(f'gradients/{name}', param.grad, global_step)
                        writer.add_scalar(f'gradient_norm/{name}', param.grad.norm(), global_step)
                for name, param in model.named_parameters():
                    writer.add_histogram(f'weights/{name}', param, global_step)
                    writer.add_scalar(f'weight_norm/{name}', param.norm(), global_step)

            optimizer.step()

            batch_loss = loss.item()
            epoch_loss += batch_loss
            batch_pbar.set_postfix({
                'batch_loss': f'{batch_loss:.4f}',
                'lr': f'{scheduler.get_last_lr()[0]:.6f}'
            })

        avg_loss = epoch_loss / config.steps_per_epoch
        train_losses.append(avg_loss)
        writer.add_scalar('Loss/train', avg_loss, epoch)

        # Validation
        model.eval()
        val_epoch_loss = 0.0
        with torch.no_grad():
            for _ in tqdm(range(config.val_steps), desc=f"Validation {epoch+1}", leave=False):
                val_inputs, val_targets = get_batch('val', config.context_length, config.batch_size, device)
                with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
                    val_logits = model(val_inputs)
                    val_loss = criterion(val_logits, val_targets)
                val_epoch_loss += val_loss.item()

        avg_val_loss = val_epoch_loss / config.val_steps
        val_losses.append(avg_val_loss)
        writer.add_scalar('Loss/val', avg_val_loss, epoch)

        # GPU memory & utilization snapshot (end of epoch)
        torch.cuda.reset_peak_memory_stats()  # reset so peak reflects this epoch only
        allocated_gb = torch.cuda.memory_allocated() / 1024**3
        peak_gb      = torch.cuda.max_memory_allocated() / 1024**3
        gpu_util     = torch.cuda.utilization()
        mem_allocated.append(allocated_gb)
        mem_peak.append(peak_gb)
        gpu_utilization.append(gpu_util)
        writer.add_scalar('GPU/memory_allocated_GB', allocated_gb, epoch)
        writer.add_scalar('GPU/memory_peak_GB', peak_gb, epoch)
        writer.add_scalar('GPU/utilization_pct', gpu_util, epoch)

        print(
            f"Epoch {epoch+1}/{config.num_epochs}, "
            f"Train Loss: {avg_loss:.4f}, "
            f"Val Loss: {avg_val_loss:.4f}, "
            f"Mem: {allocated_gb:.2f}GB (peak {peak_gb:.2f}GB), "
            f"GPU util: {gpu_util}%"
        )
        scheduler.step()

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': avg_loss,
                'val_loss': avg_val_loss,
            }, 'best_model.pt')
            print(f"Best model saved with val loss: {best_val_loss:.4f}")

    writer.close()
    print("Training complete.")
    print(torch.cuda.memory_summary())

    epochs = range(1, config.num_epochs + 1)

    fig, axes = plt.subplots(3, 1, figsize=(10, 12))

    axes[0].plot(epochs, train_losses, label='Train Loss')
    axes[0].plot(epochs, val_losses, label='Val Loss')
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].set_title("Training and Validation Loss")
    axes[0].legend()

    axes[1].plot(epochs, mem_allocated, label='Allocated (GB)')
    axes[1].plot(epochs, mem_peak, label='Peak (GB)', linestyle='--')
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Memory (GB)")
    axes[1].set_title("GPU Memory Usage")
    axes[1].legend()

    axes[2].plot(epochs, gpu_utilization, color='green', label='GPU Util %')
    axes[2].set_xlabel("Epoch")
    axes[2].set_ylabel("Utilization (%)")
    axes[2].set_title("GPU Utilization")
    axes[2].legend()

    plt.tight_layout()
    plt.savefig("training_stats.png", dpi=150, bbox_inches='tight')
    plt.show()

    model.eval()
    start_tokens = torch.tensor(
        tokenizer.encode("Once"),
        dtype=torch.long
    ).unsqueeze(0).to(device)
    generated_content = model.generate(
        start_tokens, max_new_tokens=1500
    )[0].tolist()
    print(tokenizer.decode(generated_content))
