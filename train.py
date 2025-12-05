import torch
import torch.optim as optim
from torch.amp import autocast, GradScaler
import time
from tqdm import tqdm
from colorama import Fore, Style, init

from src.config import ProjectConfig
from src.model import DeepSeekForCausalLM
from src.dataset import TextDataset
from src.monitor import MoEMicroscope
from src.utils import setup_logger, CheckpointManager

# Initialize colorama
init()

def main():
    # 1. Setup
    config = ProjectConfig()
    logger = setup_logger("DeepSeekTrainer", config.log_dir)
    
    logger.info("Initializing Data Pipeline...")
    dataset = TextDataset(config.data_path, config)
    config.vocab_size = dataset.vocab_size
    logger.info(f"Vocab Size: {config.vocab_size} | Device: {config.device}")

    # 2. Model Initialization
    logger.info("Constructing DeepSeek Architecture...")
    model = DeepSeekForCausalLM(config).to(config.device)
    
    # 3. Optimizer
    optimizer = optim.AdamW(model.parameters(), lr=config.learning_rate)
    scaler = GradScaler()
    
    # 4. Utilities
    checkpointer = CheckpointManager(config.checkpoint_dir, logger)
    monitor = MoEMicroscope(log_dir=config.log_dir)
    
    start_step = checkpointer.load(model, optimizer)
    model.train()
    
    # 5. Training Loop
    logger.info("Starting Training...") # Removed emoji to prevent UnicodeError
    
    # Create the progress bar
    pbar = tqdm(range(start_step, config.max_steps), desc="Training", unit="step")
    
    try:
        for step in pbar:
            xb, yb = dataset.get_batch()
            
            optimizer.zero_grad(set_to_none=True)
            
            # AMP Forward
            with autocast(device_type='cuda', dtype=torch.float16):
                logits, loss, router_probs = model(xb, yb)
            
            # Backward
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            
            # Monitoring
            monitor.log_step(step, router_probs)
            
            # Dashboard Update
            if step % config.log_every == 0:
                # Get L5 Entropy (Middle layer)
                mid_layer_key = f'layer_{config.num_hidden_layers//2}_entropy'
                entropy = monitor.step_data[-1].get(mid_layer_key, 0)
                
                pbar.set_postfix({
                    'Loss': f"{loss.item():.4f}",
                    'Ent': f"{entropy:.2f}",
                    'VRAM': f"{torch.cuda.memory_allocated()/1e9:.1f}GB"
                })
            
            # Checkpointing
            if step > 0 and step % config.save_every == 0:
                is_best = loss.item() < checkpointer.best_loss
                if is_best: checkpointer.best_loss = loss.item()
                checkpointer.save(model, optimizer, step, loss.item(), is_best)
                monitor.save_logs()

        # Final Save
        checkpointer.save(model, optimizer, config.max_steps, loss.item(), is_best=True)
        monitor.save_logs()
        
        # 6. Generation
        print(f"\n{Fore.GREEN}Training Complete. Generating samples...{Style.RESET_ALL}")
        model.eval()
        start_str = "\n"
        context = torch.tensor(dataset.encode(start_str), dtype=torch.long, device=config.device).unsqueeze(0)
        
        print(f"{Fore.CYAN}Generating text (this may take a moment)...{Style.RESET_ALL}")
        for _ in tqdm(range(200), desc="Generating"):
            with torch.no_grad():
                logits, _, _ = model(context)
                logits = logits[:, -1, :]
                probs = torch.nn.functional.softmax(logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
                context = torch.cat((context, next_token), dim=1)
        
        output_text = dataset.decode(context[0].tolist())
        print(f"\n{Fore.YELLOW}{'-'*20} GENERATED OUTPUT {'-'*20}{Style.RESET_ALL}")
        print(output_text)
        print(f"{Fore.YELLOW}{'-'*58}{Style.RESET_ALL}")

    except KeyboardInterrupt:
        logger.info("Training interrupted by user. Saving emergency checkpoint...")
        checkpointer.save(model, optimizer, step, loss.item())
        monitor.save_logs()
    except Exception as e:
        logger.error(f"Critical Error: {e}", exc_info=True)
        raise

if __name__ == "__main__":
    main()