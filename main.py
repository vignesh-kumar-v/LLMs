from TinyStories import TinyStoriesDataset
from CrossEntropyLoss import CrossEntropyLoss
from NanoLLM import GPT2
import tiktoken
from torch.utils.data import DataLoader
import config
import torch
torch.set_float32_matmul_precision('high')
import matplotlib.pyplot as plt
from tqdm import tqdm
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.utils.tensorboard import SummaryWriter

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Using device: {torch.cuda.get_device_name(device) if device=='cuda' else 'CPU'}")

with open("TinyStories-train.txt", 'r') as f:
    raw_text = f.read()
print("Tokenizing data...")
tokenizer = tiktoken.get_encoding("gpt2")
tokenized_data = tokenizer.encode(raw_text, allowed_special={"<|endoftext|>"})
print("Tokenization complete.")
vocab_size = tokenizer.n_vocab
del raw_text
print(f"Vocab size: {vocab_size}")

if __name__ == "__main__":
    print("Preparing dataset and dataloader...")
    dataset = TinyStoriesDataset(tokenized_data, context_length=config.context_length, stride=config.stride)
    dataloader = DataLoader(dataset, batch_size=config.batch_size, shuffle=True, drop_last=True)
    print("Dataset and dataloader ready.")
    model = GPT2(vocab_size=vocab_size, context_length=config.context_length, embed_dim=config.num_embeddings)
    model = model.to(device)
    model = torch.compile(model)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=1e-5)
    scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.num_epochs, eta_min=1e-7)
    criterion = CrossEntropyLoss()
    train_losses = []
    best_loss = float('inf')
    writer = SummaryWriter('runs/training_logs')
    print("Starting training...")
    for epoch in tqdm(range(config.num_epochs), desc="Epochs"):
        model.train()
        epoch_loss = 0.0
        num_batches = 0
        batch_pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}", leave=False)
        for batch_idx, (inputs, targets) in enumerate(batch_pbar):
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            logits = model(inputs)
            loss = criterion(logits, targets)
            loss.backward()

            if batch_idx == 0:
                global_step = epoch * len(dataloader) + batch_idx
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
            num_batches += 1
            batch_pbar.set_postfix({'batch_loss': f'{batch_loss:.4f}', 'lr': f'{scheduler.get_last_lr()[0]:.6f}'})
        avg_loss = epoch_loss / num_batches
        train_losses.append(avg_loss)
        print(f"Epoch {epoch+1}/{config.num_epochs}, Loss: {avg_loss:.4f}")
        scheduler.step()
        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': avg_loss,
            }, 'best_model.pt')
            print(f"Best model saved with loss: {best_loss:.4f}")
    writer.close()
    print("Training complete.")
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses)
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training Loss Over Epochs")
    plt.show()

    model.eval()
    start_tokens = torch.tensor(tokenizer.encode("Once"), dtype=torch.long).unsqueeze(0).to(device)
    generated_content = model.generate(start_tokens, max_new_tokens=1500)[0].tolist()
    print(tokenizer.decode(generated_content))