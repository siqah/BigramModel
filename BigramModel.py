from collections import defaultdict, Counter
import random 
import matplotlib.pyplot as plt 
import numpy as np 

class BigraModel:
    def __init__(self):
        self.bigram_counts = defaultdict(int)
        self.unigram_counts = defaultdict(int)
        
        #SPECIAL TOKENS
        self.START_TOKEN = '^'
        self.END_TOKEN = '$'

        #For Probability visualization
        self.probability_matrix = None
        self.vocab = None
    
    def train(self, text):
        """
        Train the model by counting character pairs
        Args:
            text: string of training text
        """
        text = text.lower()
        text_with_tokens = self.START_TOKEN + text + self.END_TOKEN


        print(f"trainning on{len(text)} characters....")
        print(f"Example with tokens: {text_with_tokens[:50]}...")

        #count all adjecent characters pairs 
        for i in range(len(text_with_tokens)-1):
            current_char = text_with_tokens[i]
            next_char = text_with_tokens[i+1]

            self.bigram_counts[(current_char, next_char)] +=1
            self.unigram_counts[current_char] +=1

        print(f"Found {len(self.unigram_count)} unique characters (including start/end tokens)")  
        print(f"Found {len(self.bigram_counts)} unique bigrams")  

        #Build vocabulary and probability matrix for visualization
        self.vocab = sorted(self.unigram_counts.keys())
        self._build_probability_matrix()


    def _build_probability_matrix(self): 
        """
        Convert counts to probabilities for visualization
        Creates a matrix P(next_char | current_char) for all character pairs
        """   
        #Create a matrix of zeros with dimensions (vocab_size, vocab_size)
        vocab_size = len(self.vocab)
        self.probability_matrix = np.zeros((vocab_size, vocab_size))

        
        
            

        



        
    