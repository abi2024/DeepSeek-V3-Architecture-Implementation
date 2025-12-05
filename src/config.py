from dataclasses import dataclass
import torch
import os

@dataclass
class ProjectConfig:
    # Paths
    base_dir: str = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_path: str = os.path.join(base_dir, "data", "training_data.txt")
    checkpoint_dir: str = os.path.join(base_dir, "checkpoints")
    log_dir: str = os.path.join(base_dir, "logs")

    # Model Architecture
    vocab_size: int = 49152
    hidden_size: int = 768
    # OPTIMIZATION: Reduced from 30 to 10 to fit in VRAM without Checkpointing
    num_hidden_layers: int = 10 
    num_attention_heads: int = 9
    intermediate_size: int = 1536
    max_seq_length: int = 1024 
    
    # MoE Specifics
    num_experts: int = 8
    num_shared_experts: int = 1
    top_k: int = 2
    
    # MLHA Specifics
    kv_compression_ratio: int = 8
    
    # Training
    batch_size: int = 4
    
    # CRITICAL FIX: Disable Checkpointing to prevent MoE Router Crash
    use_gradient_checkpointing: bool = False
    
    learning_rate: float = 3e-4
    max_steps: int = 10000
    save_every: int = 1000
    log_every: int = 10
    
    # Hardware
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    dtype: torch.dtype = torch.float16 if torch.cuda.is_available() else torch.float32

    def __post_init__(self):
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        os.makedirs(self.log_dir, exist_ok=True)