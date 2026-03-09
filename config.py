# Dataset
context_length = 128
batch_size = 8
steps_per_epoch = 1000
val_steps = 200

# Model
use_compile = True   # set False when profiling with ncu
num_embeddings = 128
num_heads = 4
num_blocks = 4
learning_rate = 3e-4
num_epochs = 50