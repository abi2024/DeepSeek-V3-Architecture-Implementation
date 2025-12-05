import torch
from transformers import AutoTokenizer
import logging

class TextDataset:
    def __init__(self, file_path, config):
        self.config = config
        logger = logging.getLogger("DeepSeekTrainer")
        
        # 1. Load the Real Smollm2 Tokenizer
        logger.info("Loading Smollm2 Tokenizer...")
        self.tokenizer = AutoTokenizer.from_pretrained("HuggingFaceTB/SmolLM2-135M")
        
        # 2. Expose vocab_size so train.py can see it
        self.vocab_size = self.tokenizer.vocab_size
        
        # 3. Load Text
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                text = f.read()
        except FileNotFoundError:
            raise FileNotFoundError(f"Dataset not found at: {file_path}")

        # 4. Tokenize
        # We disable the warning because we INTEND to create a sequence longer than 8192
        # We will slice it up during training.
        logger.info(f"Tokenizing {len(text)} characters...")
        tokens = self.tokenizer.encode(text, verbose=False, max_length=None) 
        
        self.data = torch.tensor(tokens, dtype=torch.long)
        logger.info(f"Dataset loaded. Total tokens: {len(self.data)}")
        
    def encode(self, s):
        return self.tokenizer.encode(s)
    
    def decode(self, l):
        return self.tokenizer.decode(l)
    
    def get_batch(self):
        # Sample a random chunk from the massive token sequence
        ix = torch.randint(len(self.data) - self.config.max_seq_length, (self.config.batch_size,))
        x = torch.stack([self.data[i:i+self.config.max_seq_length] for i in ix])
        y = torch.stack([self.data[i+1:i+self.config.max_seq_length+1] for i in ix])
        return x.to(self.config.device), y.to(self.config.device)