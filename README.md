# NanoLLM - Tiny Language Model Trainer

A lightweight GPT-2 style transformer model implementation trained on the TinyStories dataset. This project provides a minimalist approach to understanding and training transformer-based language models from scratch.

## Overview

This project implements a simplified version of GPT-2 with custom implementations of:
- Multi-head self-attention mechanism
- Token and positional embeddings
- Custom dataset loader for efficient training
- TensorBoard integration for training visualization

## Project Structure

```
LLMs/
├── main.py                    # Main training script with data preprocessing and model training
├── NanoLLM.py                 # GPT-2 model architecture with multi-head attention
├── TinyStories.py             # Custom dataset class for loading and processing text data
├── CrossEntropyLoss.py        # Custom cross-entropy loss implementation
├── config.py                  # Training hyperparameters and model configuration
├── TinyStories-train.txt      # Training dataset (not included in repo)
└── best_model.pt              # Saved model checkpoint (not included in repo)
```

## Features

- **Custom Transformer Architecture**: Implements multi-head attention with weight normalization
- **Efficient Training**: Uses strided windowing for dataset creation
- **Training Monitoring**: TensorBoard integration for visualizing loss, gradients, and weights
- **Model Checkpointing**: Automatically saves best performing model
- **Text Generation**: Includes inference capability with temperature-based sampling
- **GPU Acceleration**: CUDA support with torch.compile optimization

## Model Architecture

- **Attention Mechanism**: Multi-head self-attention with 4 heads
- **Context Length**: 128 tokens
- **Embedding Dimension**: 128
- **Vocabulary**: GPT-2 BPE tokenizer (50,257 tokens)
- **Dropout**: 0.3 for attention layers
- **Activation**: ReLU activation functions

## Requirements

- Python 3.8+
- PyTorch 2.0+
- CUDA-capable GPU (recommended)
- See `requirements.txt` for full dependencies

## Installation

1. Clone the repository:
```bash
git clone <your-repo-url>
cd LLMs
```

2. Create a virtual environment:
```bash
python -m venv .
source bin/activate  # On Windows: Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Download the TinyStories dataset:
```bash
# Place your TinyStories-train.txt in the project root
```

## Usage

### Training

Run the training script:
```bash
python main.py
```

Training parameters can be configured in `config.py`:
- `context_length`: Maximum sequence length (default: 128)
- `batch_size`: Training batch size (default: 8)
- `num_epochs`: Number of training epochs (default: 50)
- `learning_rate`: AdamW learning rate (default: 3e-4)
- `num_embeddings`: Embedding dimension (default: 128)
- `stride`: Dataset stride for overlapping windows (default: 32)

### Monitoring Training

View training metrics with TensorBoard:
```bash
tensorboard --logdir=runs
```

### Text Generation

After training, the model automatically generates text starting with "Once". To generate custom text, modify the start tokens in main.py:95:

```python
start_tokens = torch.tensor(tokenizer.encode("Your prompt here"), dtype=torch.long).unsqueeze(0).to(device)
generated_content = model.generate(start_tokens, max_new_tokens=1500)[0].tolist()
print(tokenizer.decode(generated_content))
```

## Configuration

Edit `config.py` to customize training:

```python
# Dataset
context_length = 128  # Maximum sequence length
stride = 32          # Sliding window stride
batch_size = 8       # Training batch size

# Model
num_embeddings = 128  # Embedding dimension
learning_rate = 3e-4  # Learning rate
num_epochs = 50       # Training epochs
```

## Training Details

- **Optimizer**: AdamW with weight decay (1e-5)
- **Learning Rate Schedule**: Cosine annealing (min: 1e-7)
- **Loss Function**: Custom cross-entropy implementation
- **Precision**: High precision float32 matmul
- **Compilation**: Uses torch.compile for performance

## Model Output

The trained model checkpoint (`best_model.pt`) contains:
- Model state dictionary
- Optimizer state
- Training epoch number
- Loss value

## License

MIT License

## Acknowledgments

- Built using the TinyStories dataset
- Inspired by GPT-2 architecture
- Uses OpenAI's tiktoken tokenizer

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.
