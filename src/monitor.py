import torch
import numpy as np
import os

class MoEMicroscope:
    def __init__(self, log_dir="logs"):
        self.log_dir = log_dir
        os.makedirs(log_dir, exist_ok=True)
        self.step_data = []
        
    def calculate_entropy(self, probs):
        # probs: [batch*seq, num_experts]
        # H = -sum(p * log(p))
        entropy = -torch.sum(probs * torch.log(probs + 1e-10), dim=-1).mean()
        return entropy.item()
        
    def calculate_load_balance(self, probs, num_experts):
        # Using top-1 selection for load visual
        top1_idx = torch.argmax(probs, dim=-1)
        hist = torch.bincount(top1_idx, minlength=num_experts).float()
        # Normalized load
        load_dist = (hist / hist.sum()).cpu().numpy()
        return load_dist

    def log_step(self, step, layer_probs):
        """
        layer_probs: List of tensors [B*L, num_experts] from each layer
        """
        step_metrics = {'step': step}
        
        for i, probs in enumerate(layer_probs):
            # Flatten batch/seq
            flat_probs = probs.view(-1, probs.shape[-1])
            
            # 1. Router Entropy
            h = self.calculate_entropy(flat_probs)
            step_metrics[f'layer_{i}_entropy'] = h
            
            # 2. Expert Load (for Heatmap)
            # We store the raw distribution to save to CSV
            load = self.calculate_load_balance(flat_probs, probs.shape[-1])
            step_metrics[f'layer_{i}_load'] = load.tolist()
            
            # 3. Token Dropping (Simulated)
            # DeepSeek bias-mode doesn't hard drop, but we calculate
            # what WOULD drop if capacity was 1.2x uniform.
            # capacity = (batch * seq) / num_experts * 1.2
            tokens = flat_probs.shape[0]
            num_experts = probs.shape[-1]
            capacity = int((tokens / num_experts) * 1.2)
            
            counts = torch.bincount(torch.argmax(flat_probs, dim=-1), minlength=num_experts)
            dropped = torch.sum(torch.relu(counts - capacity))
            drop_rate = dropped.item() / tokens
            step_metrics[f'layer_{i}_droprate'] = drop_rate

        self.step_data.append(step_metrics)
        
        # Simple ASCII Dashboard for Console
        if step % 100 == 0:
            print(f"\n[Microscope] Step {step}")
            print(f"L0 Entropy: {step_metrics['layer_0_entropy']:.4f} | Drop Rate: {step_metrics['layer_0_droprate']:.2%}")
            print(f"L0 Load: {['{:.2f}'.format(x) for x in step_metrics['layer_0_load']]}")

    def save_logs(self):
        import json
        with open(f"{self.log_dir}/training_metrics.json", "w") as f:
            json.dump(self.step_data, f)
        print(f"Metrics saved to {self.log_dir}/training_metrics.json")