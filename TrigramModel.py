import numpy as np
from collections import defaultdict, Counter
import matplotlib.pyplot as plt


class TrigramModel:
    def __init__(self, smoothing=1.0):
        #store raw counts
        self.trigram_counts = defaultdict(int)
        self.bigram_counts = defaultdict(int)
        self.unigram_counts = defaultdict(int)


        #special tokens
        self.START = '^'
        self.END = '$'

        #smoothing parameter
        self.smoothing = smoothing

        #for trancking vocabulary
        self.vocab = set()

        #for caching probabilities  (optimization)

        self.prob_cache = {}

    def prepare_text(self, text):
        text = text.lower()
        #add 2 start tokens at the beginning  and 1 end token at the en
        prepared = self.START + self.START + text + self.END
        return prepared

    def train(self, text, verbose=True):

        prepared = self.prepare_text(text)
        if verbose:
            print(f"\n{'='*60}")
            print(f"TRAINING TRIGRAM MODEL")
            print(f"{'='*60}")
            print(f"Original text length: {len(text)} chars")
            print(f"Prepared text length: {len(prepared)} chars")
            print(f"\nPrepared text sample: {prepared[:100]}...")


        #count trigrams by sliding window of the size
        for i in range(len(prepared)-2):
            #extract the 3 characters at position i, i+1, i+2 
            char1 = preparred[i]
            char2 = prepared[i+1]
            char3 = prepared[i+2]    


            #add to vocabulary
            self.vocab.add(char1)
            self.vocab.add(char2)
            self.vocab.add(char3)


            #Count trigram (c1,c2,c3)
            trigram_key = (char1, char2, char3)
            self.trigram_counts[trigram_key] += 1

            #Count bigram (c1,c2)
            bigram_key = (char1, char2)
            self.trigram_counts[bigram_key] += 1

            #Count unigram (ci) completeness
            self.unigram_counts[char1] += 1

        #Add END token to unigram counts         
        self.unigram_counts[self.END] = self.unigram_counts.get(self.END, 0)

        #convert vocab to sorted lidt for consistent ordering 
        self.vocab = sorted(list(self.vocab))  

        if verbose:
            print(f"\nVocabulary size: {len(self.vocab)} unique characters")
            print(f"Unique trigrams seen: {len(self.trigram_counts)}")
            print(f"Unique bigrams seen: {len(self.bigram_counts)}") 

            #show some statistics
            print(f"\nMost common characters:")
            sorted_unigram = sorted(self.unigram_counts.items(), key=lambda x: x[1], reverse=True)
            for char, count in sorted_unigram:
                if char not in [self.START, self.END]:
                    print(f"  '{char}': {count} times")

            print(f"\nMost common bigrams:")
            sorted_bigrams = sorted(self.bigram_counts.items(),
                                    key=lambda x: x[1], reverse=True )[:5]  
            for (c1, c2), count in sorted_bigrams:
                print(f"  '{c1}{c2}': {count} times")      

    def probability(self, char1, char2, char3):
        #get  count of the specific trigram 
        trigram_count = self.trigram_counts.get((char1, char2, char3), 0)

        #Get count of the conditioning bigram
        bigram_count = self.bigram_counts.get((char1, char2), 0)

        #Vocabulary size for smoothing  denominator
        vocab_size = len(self.vocab)

        #Apply Laplace smoothing
        numerator = trigram_count + self.smoothing
        denominator = bigram_count + (self.smoothing * vocab_size) 

        #handle case where bigram neverr seen (denominator = smoothing * vocab_size)
        if denominator == 0:
            return 1.0 / vocab_size #Uniform distribution

        return numerator / denominator
          
    def get_next_char_distribution(self, char1, char2):
        distribution = {}
        for char3 in self.vocab:
            prob = probability(char1, char2, char3)
            if prob > 0:
                distribution[char3] = prob

        #Normalize to ensure it sums to 1
        total = sum(distribution.values())
        if total > 0:
            distribution = {k: v/total for k, v in distribution.items()}

        return distribution

            
                




        
