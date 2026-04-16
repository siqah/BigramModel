"""
REAL TEXT TRAINING AND ANALYSIS
This script shows how to train on actual books and analyze patterns
"""

import urllib.request
import os
import numpy as np
from BigramModel import BigramModel, load_text_sample

def download_real_text():
    """
    Download a small public domain book for training
    """
    # Download "The Adventures of Sherlock Holmes" (small, ~500KB)
    url = "https://www.gutenberg.org/files/1661/1661-0.txt"
    output_file = "sherlock_holmes.txt"
    
    if not os.path.exists(output_file):
        print("Downloading Sherlock Holmes...")
        try:
            urllib.request.urlretrieve(url, output_file)
            print(f"Downloaded to {output_file}")
        except:
            print("Download failed, using sample text")
            return None
    return output_file

def analyze_model_statistics(model):
    """
    Analyze the model's learned statistics
    """
    print("\n" + "="*60)
    print("MODEL STATISTICS ANALYSIS")
    print("="*60)
    
    # Vocabulary analysis
    print(f"\nVocabulary size: {len(model.vocab)} characters")
    print("Most common characters:")
    
    # Sort characters by frequency
    char_freq = sorted(model.unigram_counts.items(), key=lambda x: x[1], reverse=True)
    for char, count in char_freq[:10]:
        if char not in [model.START_TOKEN, model.END_TOKEN]:
            print(f"  '{char}': {count} occurrences")
    
    # Entropy calculation (measure of unpredictability)
    print("\nEntropy analysis (lower = more predictable):")
    total_entropy = 0
    
    for char in model.vocab:
        if char in [model.START_TOKEN, model.END_TOKEN]:
            continue
            
        probs = model.predict_next(char)
        if probs:
            entropy = -sum(p * np.log(p) for p in probs.values())
            total_entropy += entropy * model.unigram_counts[char]
    
    total_chars = sum(model.unigram_counts.values())
    avg_entropy = total_entropy / total_chars
    print(f"  Average entropy: {avg_entropy:.3f} bits per character")
    print(f"  Perplexity: {np.exp(avg_entropy):.1f} (lower is better)")
    
    # Sparsity analysis
    total_possible_bigrams = len(model.vocab) ** 2
    observed_bigrams = len(model.bigram_counts)
    print(f"\nSparsity analysis:")
    print(f"  Possible bigrams: {total_possible_bigrams}")
    print(f"  Observed bigrams: {observed_bigrams}")
    print(f"  Coverage: {observed_bigrams/total_possible_bigrams*100:.2f}%")
    print(f"  Missing bigrams: {total_possible_bigrams - observed_bigrams}")

def visualize_chain_rule(model):
    """
    Visualize how chain rule accumulates probability
    """
    print("\n" + "="*60)
    print("CHAIN RULE VISUALIZATION")
    print("="*60)
    
    sentences = [
        "the dog ran",
        "the cat slept",
        "a quick brown fox"
    ]
    
    for sentence in sentences:
        print(f"\nSentence: '{sentence}'")
        print("-" * 40)
        
        # Calculate cumulative probability
        words = sentence.split()
        cumulative_prob = 1.0
        cumulative_log = 0
        
        # For simplicity, model characters within each word
        for word in words:
            print(f"\n  Word: '{word}'")
            prev_char = model.START_TOKEN
            
            for i, char in enumerate(word.lower()):
                prob = model.probability(prev_char, char)
                if prob == 0:
                    prob = 1e-6  # Small smoothing
                
                cumulative_prob *= prob
                cumulative_log += np.log(prob)
                
                print(f"    P('{char}'|'{prev_char}') = {prob:.6f} → "
                      f"cumulative = {cumulative_prob:.2e}")
                prev_char = char
            
            # End of word probability
            end_prob = model.probability(prev_char, ' ')
            if end_prob == 0:
                end_prob = 1e-6
            
            cumulative_prob *= end_prob
            cumulative_log += np.log(end_prob)
            print(f"    P(' '|'{prev_char}') = {end_prob:.6f} → "
                  f"cumulative = {cumulative_prob:.2e}")
        
        print(f"\n  FINAL P('{sentence}') = {cumulative_prob:.2e}")
        print(f"  Log probability = {cumulative_log:.2f}")

# Run analysis
if __name__ == "__main__":
    # Option 1: Use the downloaded text
    text_file = download_real_text()
    if text_file:
        with open(text_file, 'r', encoding='utf-8') as f:
            training_text = f.read()[:50000]  # Use first 50k chars for speed
    else:
        training_text = load_text_sample()
    
    # Train model
    model = BigramModel()
    model.train(training_text)
    
    # Analyze
    analyze_model_statistics(model)
    visualize_chain_rule(model)
