import torch
import torch.nn.functional as F
from transformers import AutoTokenizer
from src.config import ProjectConfig
from src.model import DeepSeekForCausalLM
import os
import argparse

def generate_text(model, tokenizer, prompt, max_new_tokens=200, temperature=0.7, top_k=50, device='cuda'):
    model.eval()
    
    # Encode input
    input_ids = tokenizer.encode(prompt, return_tensors='pt').to(device)
    
    # Generate
    with torch.no_grad():
        for _ in range(max_new_tokens):
            # Forward pass
            logits, _, _ = model(input_ids)
            
            # Get last token logits
            last_logits = logits[:, -1, :] / temperature
            
            # Top-K Sampling
            if top_k > 0:
                v, _ = torch.topk(last_logits, top_k)
                last_logits[last_logits < v[:, [-1]]] = -float('Inf')
            
            # Softmax & Sample
            probs = F.softmax(last_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            
            # Append
            input_ids = torch.cat((input_ids, next_token), dim=1)
            
    return tokenizer.decode(input_ids[0].tolist(), skip_special_tokens=True)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompt", type=str, default="The King said", help="Start of the text")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/best_model.pt", help="Path to model checkpoint")
    args = parser.parse_args()

    # 1. Setup
    config = ProjectConfig()
    device = config.device
    print(f"⚡ Device: {device}")
    
    # 2. Load Tokenizer (Smollm2)
    print("📖 Loading Tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained("HuggingFaceTB/SmolLM2-135M")
    config.vocab_size = tokenizer.vocab_size

    # 3. Load Model
    print(f"🏗️ Loading Model Architecture (Layers: {config.num_hidden_layers})...")
    model = DeepSeekForCausalLM(config).to(device)
    
    # 4. Load Weights
    if os.path.exists(args.checkpoint):
        print(f"💾 Loading Weights from: {args.checkpoint}")
        checkpoint = torch.load(args.checkpoint, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        loss = checkpoint.get('loss', 'N/A')
        print(f"✅ Model loaded (Best Loss: {loss})")
    else:
        print(f"❌ Checkpoint not found at {args.checkpoint}")
        return

    # 5. Generate 5 Samples
    print(f"\n{'-'*60}")
    print(f"🤖 Generating 5 Responses (Prompt: '{args.prompt}')")
    print(f"{'-'*60}")

    for i in range(5):
        output = generate_text(
            model, 
            tokenizer, 
            prompt=args.prompt, 
            max_new_tokens=150, 
            temperature=0.8
        )
        print(f"\n📝 Response {i+1}:\n{output}")
        print(f"{'-'*30}")

if __name__ == "__main__":
    main()