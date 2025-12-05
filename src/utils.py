import logging
import os
import torch
import sys
from datetime import datetime

def setup_logger(name, log_dir):
    """Sets up a production logger that writes to file and console."""
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    
    # File Handler
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    fh = logging.FileHandler(os.path.join(log_dir, f"system_{timestamp}.log"))
    fh.setFormatter(formatter)
    
    # Console Handler
    ch = logging.StreamHandler(sys.stdout)
    ch.setFormatter(formatter)
    
    logger.addHandler(fh)
    logger.addHandler(ch)
    return logger

class CheckpointManager:
    def __init__(self, checkpoint_dir, logger):
        self.dir = checkpoint_dir
        self.logger = logger
        self.best_loss = float('inf')

    def save(self, model, optimizer, step, loss, is_best=False):
        state = {
            'step': step,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': loss,
        }
        
        # Atomic saving: Write to temp, then rename
        # This prevents corrupted files if power fails during write
        tmp_path = os.path.join(self.dir, "tmp.pt")
        torch.save(state, tmp_path)
        
        # Save latest
        latest_path = os.path.join(self.dir, "latest_checkpoint.pt")
        if os.path.exists(latest_path): os.remove(latest_path)
        os.rename(tmp_path, latest_path)
        
        self.logger.info(f"Checkpoint saved: Step {step}")
        
        # Save historical checkpoint
        history_path = os.path.join(self.dir, f"checkpoint_step_{step}.pt")
        torch.save(state, history_path) # Direct save for history

        if is_best:
            best_path = os.path.join(self.dir, "best_model.pt")
            torch.save(state, best_path)
            self.logger.info(f"🎉 New best model saved! Loss: {loss:.4f}")

    def load(self, model, optimizer=None):
        path = os.path.join(self.dir, "latest_checkpoint.pt")
        if not os.path.exists(path):
            return 0  # Start from scratch
        
        self.logger.info(f"Loading checkpoint from {path}")
        checkpoint = torch.load(path)
        model.load_state_dict(checkpoint['model_state_dict'])
        if optimizer:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        return checkpoint['step']