import torch
import torch.nn.functional as F
from transformers import AutoTokenizer
from src.config import ProjectConfig
from src.model import DeepSeekForCausalLM
import os
import argparse
import time
import json
from collections import Counter

# ============================================================================
# Generation Utilities
# ============================================================================

def top_k_top_p_filtering(logits, top_k=50, top_p=0.95, filter_value=-float('Inf')):
    """Apply top-k and/or top-p (nucleus) filtering to logits."""
    top_k = min(top_k, logits.size(-1))
    
    # Top-K filtering
    if top_k > 0:
        indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
        logits[indices_to_remove] = filter_value

    # Top-P (nucleus) filtering
    if top_p < 1.0:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
        
        # Remove tokens with cumulative probability above threshold
        sorted_indices_to_remove = cumulative_probs > top_p
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0
        
        indices_to_remove = sorted_indices_to_remove.scatter(
            dim=-1, index=sorted_indices, src=sorted_indices_to_remove
        )
        logits[indices_to_remove] = filter_value

    return logits


def generate_text(
    model, 
    tokenizer, 
    prompt, 
    max_new_tokens=200, 
    temperature=0.7, 
    top_k=50, 
    top_p=0.95,
    repetition_penalty=1.1,
    device='cuda',
    return_stats=False
):
    """Generate text with advanced sampling options."""
    model.eval()
    
    input_ids = tokenizer.encode(prompt, return_tensors='pt').to(device)
    prompt_length = input_ids.shape[1]
    generated_tokens = []
    
    start_time = time.perf_counter()
    
    with torch.no_grad():
        for _ in range(max_new_tokens):
            logits, _, _ = model(input_ids)
            next_token_logits = logits[:, -1, :].clone()
            
            # Apply repetition penalty
            if repetition_penalty != 1.0:
                for token_id in set(input_ids[0].tolist()):
                    next_token_logits[0, token_id] /= repetition_penalty
            
            # Apply temperature
            next_token_logits = next_token_logits / temperature
            
            # Apply top-k and top-p filtering
            filtered_logits = top_k_top_p_filtering(
                next_token_logits, top_k=top_k, top_p=top_p
            )
            
            # Sample
            probs = F.softmax(filtered_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            
            generated_tokens.append(next_token.item())
            input_ids = torch.cat((input_ids, next_token), dim=1)
            
            # Stop on EOS
            if next_token.item() == tokenizer.eos_token_id:
                break
    
    elapsed = time.perf_counter() - start_time
    output_text = tokenizer.decode(input_ids[0].tolist(), skip_special_tokens=True)
    
    if return_stats:
        stats = {
            'tokens_generated': len(generated_tokens),
            'time_seconds': elapsed,
            'tokens_per_second': len(generated_tokens) / elapsed if elapsed > 0 else 0,
            'prompt_tokens': prompt_length,
        }
        return output_text, stats
    
    return output_text


def calculate_perplexity(model, tokenizer, text, device='cuda'):
    """Calculate perplexity of the model on given text."""
    model.eval()
    input_ids = tokenizer.encode(text, return_tensors='pt').to(device)
    
    with torch.no_grad():
        logits, _, _ = model(input_ids)
        # Shift for next-token prediction
        shift_logits = logits[:, :-1, :].contiguous()
        shift_labels = input_ids[:, 1:].contiguous()
        
        loss = F.cross_entropy(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1),
            reduction='mean'
        )
        perplexity = torch.exp(loss).item()
    
    return perplexity


def analyze_repetition(text, n_grams=[2, 3, 4]):
    """Analyze n-gram repetition in generated text."""
    words = text.lower().split()
    results = {}
    
    for n in n_grams:
        if len(words) < n:
            results[f'{n}-gram'] = {'total': 0, 'unique': 0, 'repetition_rate': 0}
            continue
            
        ngrams = [tuple(words[i:i+n]) for i in range(len(words) - n + 1)]
        total = len(ngrams)
        unique = len(set(ngrams))
        repetition_rate = 1 - (unique / total) if total > 0 else 0
        
        results[f'{n}-gram'] = {
            'total': total,
            'unique': unique,
            'repetition_rate': round(repetition_rate * 100, 2)
        }
    
    return results


# ============================================================================
# Test Suites
# ============================================================================

TEST_PROMPTS = {
    'narrative': [
        "The King said",
        "In the country",
        "Once upon a time",
        "The brave knight",
        "In a distant land",
    ],
    'dialogue': [
        "ROMEO:",
        "HAMLET:\nTo be",
        "KING RICHARD:\nNow is",
        "JULIET:\nO Romeo",
        "MACBETH:\nIs this",
    ],
    'descriptive': [
        "The castle stood",
        "A beautiful morning",
        "The dark forest",
        "Upon the hill",
        "The ancient temple",
    ],
    'action': [
        "He drew his sword",
        "She ran through",
        "They fought bravely",
        "The battle began",
        "With a mighty blow",
    ],
}

TEMPERATURE_TESTS = [0.3, 0.5, 0.7, 1.0, 1.2]

SAMPLING_CONFIGS = [
    {'name': 'Greedy-ish', 'temperature': 0.3, 'top_k': 10, 'top_p': 1.0},
    {'name': 'Balanced', 'temperature': 0.7, 'top_k': 50, 'top_p': 0.95},
    {'name': 'Creative', 'temperature': 1.0, 'top_k': 100, 'top_p': 0.9},
    {'name': 'Nucleus Only', 'temperature': 0.8, 'top_k': 0, 'top_p': 0.92},
    {'name': 'High Temp', 'temperature': 1.2, 'top_k': 50, 'top_p': 0.95},
]


def run_prompt_suite(model, tokenizer, device, max_tokens=150):
    """Test model across different prompt categories."""
    print("\n" + "="*70)
    print("📋 PROMPT CATEGORY TEST SUITE")
    print("="*70)
    
    results = {}
    
    for category, prompts in TEST_PROMPTS.items():
        print(f"\n{'─'*70}")
        print(f"📁 Category: {category.upper()}")
        print(f"{'─'*70}")
        
        category_results = []
        
        for prompt in prompts:
            output, stats = generate_text(
                model, tokenizer, prompt,
                max_new_tokens=max_tokens,
                temperature=0.7,
                device=device,
                return_stats=True
            )
            
            rep_analysis = analyze_repetition(output)
            
            result = {
                'prompt': prompt,
                'output': output,
                'stats': stats,
                'repetition': rep_analysis
            }
            category_results.append(result)
            
            print(f"\n🎯 Prompt: \"{prompt}\"")
            print(f"📝 Output ({stats['tokens_generated']} tokens, {stats['tokens_per_second']:.1f} tok/s):")
            print(f"   {output[:300]}{'...' if len(output) > 300 else ''}")
            print(f"   Repetition: 2-gram={rep_analysis['2-gram']['repetition_rate']}%, "
                  f"3-gram={rep_analysis['3-gram']['repetition_rate']}%")
        
        results[category] = category_results
    
    return results


def run_temperature_test(model, tokenizer, device, prompt="The King said"):
    """Test same prompt across different temperatures."""
    print("\n" + "="*70)
    print("🌡️  TEMPERATURE SWEEP TEST")
    print(f"    Prompt: \"{prompt}\"")
    print("="*70)
    
    results = []
    
    for temp in TEMPERATURE_TESTS:
        output, stats = generate_text(
            model, tokenizer, prompt,
            max_new_tokens=100,
            temperature=temp,
            device=device,
            return_stats=True
        )
        
        rep = analyze_repetition(output)
        
        print(f"\n🌡️  Temperature: {temp}")
        print(f"   {output[:250]}{'...' if len(output) > 250 else ''}")
        print(f"   Stats: {stats['tokens_per_second']:.1f} tok/s | "
              f"Rep: {rep['3-gram']['repetition_rate']}%")
        
        results.append({
            'temperature': temp,
            'output': output,
            'stats': stats,
            'repetition': rep
        })
    
    return results


def run_sampling_test(model, tokenizer, device, prompt="Once upon a time"):
    """Test different sampling configurations."""
    print("\n" + "="*70)
    print("🎲 SAMPLING STRATEGY TEST")
    print(f"   Prompt: \"{prompt}\"")
    print("="*70)
    
    results = []
    
    for config in SAMPLING_CONFIGS:
        output, stats = generate_text(
            model, tokenizer, prompt,
            max_new_tokens=100,
            temperature=config['temperature'],
            top_k=config['top_k'],
            top_p=config['top_p'],
            device=device,
            return_stats=True
        )
        
        rep = analyze_repetition(output)
        
        print(f"\n🎲 {config['name']} (T={config['temperature']}, K={config['top_k']}, P={config['top_p']})")
        print(f"   {output[:250]}{'...' if len(output) > 250 else ''}")
        print(f"   Rep: {rep['3-gram']['repetition_rate']}%")
        
        results.append({
            'config': config,
            'output': output,
            'stats': stats,
            'repetition': rep
        })
    
    return results


def run_perplexity_test(model, tokenizer, device):
    """Test perplexity on sample texts."""
    print("\n" + "="*70)
    print("📊 PERPLEXITY EVALUATION")
    print("="*70)
    
    test_texts = [
        "The king spoke to his loyal subjects about the coming war.",
        "To be or not to be, that is the question.",
        "In the beginning, there was nothing but darkness and silence.",
        "The quick brown fox jumps over the lazy dog.",
        "Once upon a time in a kingdom far away lived a princess.",
    ]
    
    results = []
    
    for text in test_texts:
        ppl = calculate_perplexity(model, tokenizer, text, device)
        results.append({'text': text, 'perplexity': ppl})
        print(f"\n📝 \"{text[:50]}...\"")
        print(f"   Perplexity: {ppl:.2f}")
    
    avg_ppl = sum(r['perplexity'] for r in results) / len(results)
    print(f"\n{'─'*70}")
    print(f"📈 Average Perplexity: {avg_ppl:.2f}")
    
    return results, avg_ppl


def run_consistency_test(model, tokenizer, device, prompt="The King said", n_runs=5):
    """Test output diversity with same prompt (same settings)."""
    print("\n" + "="*70)
    print("🔄 CONSISTENCY/DIVERSITY TEST")
    print(f"   Prompt: \"{prompt}\" × {n_runs} runs")
    print("="*70)
    
    outputs = []
    
    for i in range(n_runs):
        output = generate_text(
            model, tokenizer, prompt,
            max_new_tokens=80,
            temperature=0.8,
            device=device
        )
        outputs.append(output)
        print(f"\n   Run {i+1}: {output[:150]}...")
    
    # Calculate diversity (unique first 20 words)
    first_words = [' '.join(o.split()[:20]) for o in outputs]
    unique_starts = len(set(first_words))
    
    print(f"\n{'─'*70}")
    print(f"🎯 Diversity Score: {unique_starts}/{n_runs} unique openings")
    
    return outputs, unique_starts / n_runs


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="Comprehensive Model Testing Suite")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/best_model.pt")
    parser.add_argument("--test", type=str, default="all",
                        choices=['all', 'prompts', 'temperature', 'sampling', 'perplexity', 'consistency', 'quick'],
                        help="Which test suite to run")
    parser.add_argument("--prompt", type=str, default=None, help="Custom prompt for single test")
    parser.add_argument("--max_tokens", type=int, default=150)
    parser.add_argument("--save_results", action='store_true', help="Save results to JSON")
    args = parser.parse_args()

    # Setup
    config = ProjectConfig()
    device = config.device
    print(f"⚡ Device: {device}")
    
    # Load Tokenizer
    print("📖 Loading Tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained("HuggingFaceTB/SmolLM2-135M")
    config.vocab_size = tokenizer.vocab_size

    # Load Model
    print(f"🏗️  Loading Model...")
    model = DeepSeekForCausalLM(config).to(device)
    
    if os.path.exists(args.checkpoint):
        print(f"💾 Loading: {args.checkpoint}")
        checkpoint = torch.load(args.checkpoint, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        step = checkpoint.get('step', '?')
        loss = checkpoint.get('loss', 'N/A')
        print(f"✅ Loaded (Step: {step}, Loss: {loss:.4f})")
    else:
        print(f"❌ Checkpoint not found: {args.checkpoint}")
        return

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    print(f"📊 Total Parameters: {total_params:,}")

    all_results = {'checkpoint': args.checkpoint}

    # Run tests
    if args.test in ['all', 'quick', 'prompts']:
        prompts = TEST_PROMPTS if args.test != 'quick' else {'narrative': TEST_PROMPTS['narrative'][:2]}
        all_results['prompts'] = run_prompt_suite(model, tokenizer, device, args.max_tokens)

    if args.test in ['all', 'temperature']:
        prompt = args.prompt or "The King said"
        all_results['temperature'] = run_temperature_test(model, tokenizer, device, prompt)

    if args.test in ['all', 'sampling']:
        prompt = args.prompt or "Once upon a time"
        all_results['sampling'] = run_sampling_test(model, tokenizer, device, prompt)

    if args.test in ['all', 'perplexity']:
        results, avg = run_perplexity_test(model, tokenizer, device)
        all_results['perplexity'] = {'results': results, 'average': avg}

    if args.test in ['all', 'consistency']:
        prompt = args.prompt or "The King said"
        outputs, diversity = run_consistency_test(model, tokenizer, device, prompt)
        all_results['consistency'] = {'outputs': outputs, 'diversity_score': diversity}

    # Save results
    if args.save_results:
        output_file = 'logs/test_results.json'
        # Convert non-serializable items
        def clean_for_json(obj):
            if isinstance(obj, dict):
                return {k: clean_for_json(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [clean_for_json(i) for i in obj]
            elif isinstance(obj, float):
                return round(obj, 4)
            return obj
        
        with open(output_file, 'w') as f:
            json.dump(clean_for_json(all_results), f, indent=2)
        print(f"\n💾 Results saved to {output_file}")

    print("\n" + "="*70)
    print("✅ TESTING COMPLETE")
    print("="*70)


if __name__ == "__main__":
    main()