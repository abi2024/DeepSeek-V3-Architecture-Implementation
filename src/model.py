import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from torch.utils.checkpoint import checkpoint
from src.config import ProjectConfig

class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        norm_x = torch.mean(x * x, dim=-1, keepdim=True)
        x_normed = x * torch.rsqrt(norm_x + self.eps)
        return self.weight * x_normed

class DeepSeekMLHA(nn.Module):
    def __init__(self, config: ProjectConfig):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = config.hidden_size // config.num_attention_heads
        self.attention_dim = self.num_heads * self.head_dim
        self.kv_lora_rank = config.hidden_size // config.kv_compression_ratio
        
        self.q_proj = nn.Linear(self.hidden_size, self.attention_dim, bias=False)
        self.kv_down_proj = nn.Linear(self.hidden_size, self.kv_lora_rank, bias=False)
        self.kv_norm = RMSNorm(self.kv_lora_rank)
        self.kv_up_proj = nn.Linear(self.kv_lora_rank, 2 * self.attention_dim, bias=False)
        self.o_proj = nn.Linear(self.attention_dim, self.hidden_size, bias=False)

    def forward(self, x):
        B, L, D = x.shape
        q = self.q_proj(x).view(B, L, self.num_heads, self.head_dim)
        latent_kv = self.kv_down_proj(x)
        latent_kv = self.kv_norm(latent_kv)
        kv = self.kv_up_proj(latent_kv)
        k, v = kv.chunk(2, dim=-1)
        k = k.view(B, L, self.num_heads, self.head_dim)
        v = v.view(B, L, self.num_heads, self.head_dim)

        scores = torch.einsum('bxhd,byhd->bhxy', q, k) / math.sqrt(self.head_dim)
        mask = torch.triu(torch.ones(L, L, device=x.device) * float('-inf'), diagonal=1)
        scores = scores + mask
        
        attn = F.softmax(scores, dim=-1)
        out = torch.einsum('bhxy,byhd->bxhd', attn, v)
        return self.o_proj(out.reshape(B, L, self.attention_dim))

class Expert(nn.Module):
    def __init__(self, config: ProjectConfig):
        super().__init__()
        self.w1 = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.w2 = nn.Linear(config.intermediate_size, config.hidden_size, bias=False)
        self.w3 = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)

    def forward(self, x):
        return self.w2(F.silu(self.w1(x)) * self.w3(x))

class DeepSeekRouter(nn.Module):
    def __init__(self, config: ProjectConfig):
        super().__init__()
        self.num_experts = config.num_experts
        self.top_k = config.top_k
        self.gate = nn.Linear(config.hidden_size, config.num_experts, bias=False)
        
        # Jitter for Iteration 2
        self.jitter_noise = config.router_jitter_noise
        
        # Loss-less Balancing
        self.register_buffer('expert_bias', torch.zeros(config.num_experts))
        
        # Configurable Update Rate (Iteration 2 change)
        self.bias_update_rate = config.expert_bias_update_rate 

    def forward(self, x, training=True):
        # x: [B, L, D] -> Flatten to [B*L, D]
        flat_x = x.view(-1, x.shape[-1])
        
        # 1. Compute Raw Logits
        logits = self.gate(flat_x)
        
        # 2. Add Jitter (Exploration) - Only during training
        if training and self.jitter_noise > 0:
            noise = torch.randn_like(logits) * self.jitter_noise
            logits = logits + noise
            
        # 3. Add Bias (Load Balancing)
        logits = logits + self.expert_bias
        
        probs = F.softmax(logits, dim=-1)
        
        # Top-K
        top_k_weights, top_k_indices = torch.topk(probs, self.top_k, dim=-1)
        top_k_weights = top_k_weights / top_k_weights.sum(dim=-1, keepdim=True)
        
        # Update Bias
        if training:
            with torch.no_grad():
                expert_mask = F.one_hot(top_k_indices, num_classes=self.num_experts).sum(dim=1)
                load = expert_mask.sum(0) / expert_mask.sum()
                target_load = 1.0 / self.num_experts
                
                # If load > target, decrease bias
                error = load - target_load
                self.expert_bias -= self.bias_update_rate * torch.sign(error)
        
        return top_k_weights, top_k_indices, probs

class DeepSeekMoELayer(nn.Module):
    def __init__(self, config: ProjectConfig):
        super().__init__()
        self.num_shared = config.num_shared_experts
        self.shared_experts = nn.ModuleList([Expert(config) for _ in range(self.num_shared)])
        self.routed_experts = nn.ModuleList([Expert(config) for _ in range(config.num_experts)])
        self.router = DeepSeekRouter(config)
        
    def forward(self, x):
        B, L, D = x.shape
        shared_out = sum(exp(x) for exp in self.shared_experts)
        weights, indices, full_probs = self.router(x, self.training)
        
        flat_x = x.view(-1, D)
        flat_out = torch.zeros_like(flat_x)
        
        for k in range(self.router.top_k):
            idx_k = indices[:, k]
            w_k = weights[:, k]
            for expert_idx in range(len(self.routed_experts)):
                mask = (idx_k == expert_idx)
                if mask.any():
                    selected_x = flat_x[mask]
                    expert_out = self.routed_experts[expert_idx](selected_x)
                    flat_out[mask] += w_k[mask].unsqueeze(-1) * expert_out
        
        return shared_out + flat_out.view(B, L, D), full_probs

class DeepSeekBlock(nn.Module):
    def __init__(self, config: ProjectConfig):
        super().__init__()
        self.input_layernorm = RMSNorm(config.hidden_size)
        self.self_attn = DeepSeekMLHA(config)
        self.post_attention_layernorm = RMSNorm(config.hidden_size)
        self.mlp = DeepSeekMoELayer(config)

    def forward(self, x):
        h = x + self.self_attn(self.input_layernorm(x))
        moe_out, router_probs = self.mlp(self.post_attention_layernorm(h))
        return h + moe_out, router_probs

class DeepSeekForCausalLM(nn.Module):
    def __init__(self, config: ProjectConfig):
        super().__init__()
        self.config = config
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = nn.ModuleList([DeepSeekBlock(config) for _ in range(config.num_hidden_layers)])
        self.norm = RMSNorm(config.hidden_size)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.lm_head.weight = self.embed_tokens.weight

    def forward(self, input_ids, targets=None):
        x = self.embed_tokens(input_ids)
        all_router_probs = []
        
        for layer in self.layers:
            if self.config.use_gradient_checkpointing and self.training:
                # Gradient Checkpointing: saves memory by recomputing activations during backward pass
                x, probs = checkpoint(layer, x, use_reentrant=False)
            else:
                x, probs = layer(x)
            all_router_probs.append(probs)
            
        x = self.norm(x)
        logits = self.lm_head(x)
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        return logits, loss, all_router_probs