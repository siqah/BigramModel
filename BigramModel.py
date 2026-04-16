from collections import defaultdict, Counter
import random 
import matplotlib.pyplot as plt 
import numpy as np 

class BigramModel:
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


        print(f"Training on {len(text)} characters....")
        print(f"Example with tokens: {text_with_tokens[:50]}...")

        #count all adjecent characters pairs 
        for i in range(len(text_with_tokens)-1):
            current_char = text_with_tokens[i]
            next_char = text_with_tokens[i+1]

            self.bigram_counts[(current_char, next_char)] +=1
            self.unigram_counts[current_char] +=1

        print(f"Found {len(self.unigram_counts)} unique characters (including start/end tokens)")  
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

        for i, current_char in enumerate(self.vocab):
            for j, next_char in enumerate(self.vocab):
                if self.unigram_counts[current_char] > 0:
                    prob = self.bigram_counts[(current_char, next_char)] / self.unigram_counts[current_char]
                    self.probability_matrix[i][j] = prob

    def generate_text(self, length=100, seed=None):
        """
        Generate text based on learned bigram model
        """
        if seed is not None:
            np.random.seed(seed)
            
        if self.probability_matrix is None:
            return "Model not trained yet."

        current_char = self.START_TOKEN
        generated_text = ""

        while len(generated_text) < length:
            try:
                curr_idx = self.vocab.index(current_char)
            except ValueError:
                break
                
            probs = self.probability_matrix[curr_idx]
            
            prob_sum = np.sum(probs)
            if prob_sum == 0:
                break
                
            probs = probs / prob_sum
                
            next_char = np.random.choice(self.vocab, p=probs)
            
            if next_char == self.END_TOKEN:
                break
                
            generated_text += next_char
            current_char = next_char
            
        return generated_text
        
    def plot_matrix(self):
        """
        Visualizes the probability matrix using matplotlib
        """
        if self.probability_matrix is None:
            print("Train the model first!")
            return
            
        plt.figure(figsize=(10, 8))
        plt.imshow(self.probability_matrix, cmap="Blues")
        plt.colorbar(label="Probability")
        plt.xticks(np.arange(len(self.vocab)), self.vocab)
        plt.yticks(np.arange(len(self.vocab)), self.vocab)
        plt.title("Bigram Transition Probabilities")
        plt.xlabel("Next character")
        plt.ylabel("Current character")
        plt.tight_layout()
        # plt.show() # Commented out to avoid blocking execution in terminal

if __name__ == "__main__":
    # Example usage
    sample_text = "to be or not to be that is the question whether tis nobler in the mind to suffer the slings and arrows of outrageous fortune or to take arms against a sea of troubles and by opposing end them"
    
    model = BigramModel()
    model.train(sample_text)
    
    print("\n--- Generated Text ---")
    for i in range(5):
        print(f"Sample {i+1}: {model.generate_text(length=50)}")
