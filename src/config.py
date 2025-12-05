from dataclasses import dataclass
import torch
import os

@dataclass
class ProjectConfig:
    # ... [Keep existing path settings] ...
    base_dir: str = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_path: str = os.path.join(base_dir, "data", "training_data.txt")
    checkpoint_dir: str = os.path.join(base_dir, "checkpoints")
    log_dir: str = os.path.join(base_dir, "logs")

    # Architecture
    vocab_size: int = 49152
    hidden_size: int = 768
    num_hidden_layers: int = 10 
    num_attention_heads: int = 9
    intermediate_size: int = 1536
    max_seq_length: int = 1024 
    
    # MoE & MLHA
    num_experts: int = 8
    num_shared_experts: int = 1
    top_k: int = 2
    kv_compression_ratio: int = 8
    
    # --- ITERATION 2 UPDATES ---
    # 1. Increased Balancing Rate (Was hardcoded 0.001)
    expert_bias_update_rate: float = 0.05
    
    # 2. Added Router Jitter (New)
    router_jitter_noise: float = 0.02
    # ---------------------------

    # Training
    batch_size: int = 4
    use_gradient_checkpointing: bool = False
    
    # 3. Lower Learning Rate for Stability (Was 3e-4)
    learning_rate: float = 2e-4
    
    max_steps: int = 10000
    save_every: int = 1000
    log_every: int = 10
    
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    dtype: torch.dtype = torch.float16 if torch.cuda.is_available() else torch.float32

    def __post_init__(self):
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        os.makedirs(self.log_dir, exist_ok=True)