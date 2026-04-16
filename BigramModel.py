"""
BIGRAM LANGUAGE MODEL - Complete Implementation
This model learns character-level patterns using the probability chain rule
P(word) = P(c1) * P(c2|c1) * P(c3|c1,c2) * ... but simplified to P(c2|c1) only
"""

from collections import defaultdict, Counter
import random
import matplotlib.pyplot as plt
import numpy as np

class BigramModel:
    """
    A character-level bigram model that learns P(next_char | current_char)
    Uses maximum likelihood estimation (counting) to compute probabilities
    """
    
    def __init__(self):
        # Store counts: bigram_counts[(char1, char2)] = frequency
        self.bigram_counts = defaultdict(int)
        
        # Store counts: unigram_counts[char] = frequency
        self.unigram_counts = defaultdict(int)
        
        # Special tokens
        self.START_TOKEN = '^'  # Marks beginning of text
        self.END_TOKEN = '$'    # Marks end of text
        
        # For probability visualization
        self.probability_matrix = None
        self.vocab = None
    
    def train(self, text):
        """
        Train the model by counting character pairs
        Args:
            text: string of training text
        """
        # Clean and prepare text (lowercase for simplicity)
        text = text.lower()
        
        # Add start token at beginning and end token at end
        # This helps model learn how to start and end sentences
        text_with_tokens = self.START_TOKEN + text + self.END_TOKEN
        
        print(f"Training on {len(text)} characters...")
        print(f"Example with tokens: {text_with_tokens[:50]}...")
        
        # Count all adjacent character pairs
        for i in range(len(text_with_tokens) - 1):
            current_char = text_with_tokens[i]
            next_char = text_with_tokens[i + 1]
            
            # Increment counts
            self.bigram_counts[(current_char, next_char)] += 1
            self.unigram_counts[current_char] += 1
        
        print(f"Found {len(self.unigram_counts)} unique characters (including tokens)")
        print(f"Found {len(self.bigram_counts)} unique bigrams")
        
        # Build vocabulary and probability matrix for visualization
        self.vocab = sorted(list(self.unigram_counts.keys()))
        self._build_probability_matrix()
    
    def _build_probability_matrix(self):
        """
        Convert counts to probabilities for visualization
        Creates a matrix P(next_char | current_char) for all character pairs
        """
        vocab_size = len(self.vocab)
        self.probability_matrix = np.zeros((vocab_size, vocab_size))
        
        for i, char1 in enumerate(self.vocab):
            total = self.unigram_counts[char1]
            if total == 0:
                continue
            
            for j, char2 in enumerate(self.vocab):
                count = self.bigram_counts.get((char1, char2), 0)
                self.probability_matrix[i, j] = count / total
    
    def probability(self, char1, char2):
        """
        Calculate P(char2 | char1)
        Args:
            char1: first character
            char2: second character
        Returns:
            probability (float)
        """
        if self.unigram_counts[char1] == 0:
            return 0.0
        return self.bigram_counts.get((char1, char2), 0) / self.unigram_counts[char1]
    
    def predict_next(self, current_char):
        """
        Get probability distribution for next character
        Args:
            current_char: the conditioning character
        Returns:
            dictionary of {char: probability}
        """
        if current_char not in self.unigram_counts:
            return {}
        
        total = self.unigram_counts[current_char]
        probs = {}
        
        for (c1, c2), count in self.bigram_counts.items():
            if c1 == current_char:
                probs[c2] = count / total
        
        return probs
    
    def generate(self, max_length=100, temperature=1.0):
        """
        Generate text using the chain rule and sampling
        Args:
            max_length: maximum characters to generate
            temperature: controls randomness (0=deterministic, 1=normal, >1=more random)
        Returns:
            generated string
        """
        current_char = self.START_TOKEN
        result = []
        
        for _ in range(max_length):
            # Get probability distribution for next character
            probs_dict = self.predict_next(current_char)
            
            if not probs_dict:
                break
            
            # Convert to lists for sampling
            chars = list(probs_dict.keys())
            probs = np.array(list(probs_dict.values()))
            
            # Apply temperature scaling
            if temperature != 1.0:
                # Lower temperature = sharper peaks (more deterministic)
                # Higher temperature = flatter distribution (more random)
                probs = np.log(probs + 1e-10) / temperature
                probs = np.exp(probs)
                probs = probs / probs.sum()
            
            # Sample next character from distribution
            next_char = np.random.choice(chars, p=probs)
            
            # Stop if we hit end token
            if next_char == self.END_TOKEN:
                break
            
            # Add to result
            result.append(next_char)
            current_char = next_char
        
        return ''.join(result)
    
    def calculate_sentence_probability(self, sentence):
        """
        Apply the chain rule to compute P(sentence)
        P(s1, s2, ..., sn) = P(s1) * P(s2|s1) * ... * P(sn|s1...sn-1)
        For bigram model, we approximate P(si|all previous) ≈ P(si|si-1)
        
        Args:
            sentence: string to evaluate
        Returns:
            log probability (using logs to avoid underflow)
        """
        sentence = sentence.lower()
        log_prob = 0.0
        
        # Add start token at beginning
        prev_char = self.START_TOKEN
        
        for i, char in enumerate(sentence):
            prob = self.probability(prev_char, char)
            if prob == 0:
                # Laplace smoothing: add small probability for unseen pairs
                prob = 1e-10
            log_prob += np.log(prob)
            prev_char = char
        
        # Add probability of end token
        end_prob = self.probability(prev_char, self.END_TOKEN)
        if end_prob == 0:
            end_prob = 1e-10
        log_prob += np.log(end_prob)
        
        return log_prob
    
    def visualize_probabilities(self, top_n=10):
        """
        Visualize the most probable character pairs
        """
        print("\n" + "="*50)
        print("TOP 10 MOST PROBABLE BIGRAMS")
        print("="*50)
        
        # Sort bigrams by probability
        bigram_probs = []
        for (c1, c2), count in self.bigram_counts.items():
            prob = count / self.unigram_counts[c1]
            bigram_probs.append((c1, c2, prob))
        
        bigram_probs.sort(key=lambda x: x[2], reverse=True)
        
        for i, (c1, c2, prob) in enumerate(bigram_probs[:top_n], 1):
            print(f"{i:2d}. '{c1}' → '{c2}': {prob:.4f} ({prob*100:.2f}%)")
        
        print("\n" + "="*50)
        print("INTERESTING PROBABILITIES")
        print("="*50)
        
        # Show some interesting examples
        examples = [(' ', 't'), ('t', 'h'), ('h', 'e'), ('e', ' '), 
                   ('.', ' '), ('?', ' '), ('!', ' ')]
        
        for c1, c2 in examples:
            prob = self.probability(c1, c2)
            print(f"P('{c2}' | '{c1}') = {prob:.4f} ({prob*100:.2f}%)")

def load_text_sample():
    """
    Load a sample text for training
    You can replace this with any text file
    """
    # Sample text - you can replace with actual file reading
    sample_text = """
    The quick brown fox jumps over the lazy dog. This sentence contains every letter of the alphabet.
    Machine learning is fascinating. We can build models that learn patterns from data.
    The probability chain rule is fundamental to language modeling.
    Each word depends on previous words in a sequence.
    """
    
    return sample_text

# ============================================================================
# DEMONSTRATION AND TESTING
# ============================================================================

if __name__ == "__main__":
    print("="*60)
    print("BIGRAM LANGUAGE MODEL DEMONSTRATION")
    print("="*60)
    
    # 1. Load and train
    print("\n[1] Loading training data...")
    training_text = load_text_sample()
    print(f"Training text length: {len(training_text)} characters")
    print(f"Sample: {training_text[:200]}...")
    
    print("\n[2] Training bigram model...")
    model = BigramModel()
    model.train(training_text)
    
    # 2. Visualize learned probabilities
    model.visualize_probabilities()
    
    # 3. Generate text with different temperatures
    print("\n" + "="*50)
    print("TEXT GENERATION (Temperature=0.5 - Conservative)")
    print("="*50)
    for i in range(3):
        generated = model.generate(max_length=100, temperature=0.5)
        print(f"{i+1}. {generated}")
    
    print("\n" + "="*50)
    print("TEXT GENERATION (Temperature=1.0 - Normal)")
    print("="*50)
    for i in range(3):
        generated = model.generate(max_length=100, temperature=1.0)
        print(f"{i+1}. {generated}")
    
    print("\n" + "="*50)
    print("TEXT GENERATION (Temperature=2.0 - Random)")
    print("="*50)
    for i in range(3):
        generated = model.generate(max_length=100, temperature=2.0)
        print(f"{i+1}. {generated}")
    
    # 4. Calculate sentence probabilities using chain rule
    print("\n" + "="*50)
    print("SENTENCE PROBABILITIES (Using Chain Rule)")
    print("="*50)
    
    test_sentences = [
        "the quick brown fox",
        "the lazy dog sleeps",
        "machine learning is fun",
        "zzz xyz abc def"  # Unlikely sequence
    ]
    
    for sentence in test_sentences:
        log_prob = model.calculate_sentence_probability(sentence)
        prob = np.exp(log_prob)  # Convert back from log
        print(f"P('{sentence}') = {prob:.2e} (log: {log_prob:.2f})")
    
    # 5. Demonstrate the chain rule step-by-step
    print("\n" + "="*50)
    print("CHAIN RULE DEMONSTRATION")
    print("="*50)
    
    example_sentence = "the cat"
    print(f"Calculating P('{example_sentence}') step by step:")
    print()
    
    example_sentence = example_sentence.lower()
    prev_char = model.START_TOKEN
    total_log_prob = 0
    
    for i, char in enumerate(example_sentence):
        prob = model.probability(prev_char, char)
        print(f"Step {i+1}: P('{char}' | '{prev_char}') = {prob:.4f}")
        if prob > 0:
            print(f"         {model.unigram_counts[prev_char]} occurrences of '{prev_char}'")
            print(f"         {model.bigram_counts.get((prev_char, char), 0)} occurrences of '{prev_char}{char}'")
        total_log_prob += np.log(prob + 1e-10)
        prev_char = char
    
    # Add end token
    end_prob = model.probability(prev_char, model.END_TOKEN)
    print(f"Step {len(example_sentence)+1}: P('{model.END_TOKEN}' | '{prev_char}') = {end_prob:.4f}")
    total_log_prob += np.log(end_prob + 1e-10)
    
    print(f"\nFinal log probability: {total_log_prob:.2f}")
    print(f"Final probability: {np.exp(total_log_prob):.2e}")
