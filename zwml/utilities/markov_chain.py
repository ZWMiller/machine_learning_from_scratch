import numpy as np

class markov_chain:
    
    def __init__(self, text, from_file=True, ngram=2, random_state=None):
        """
        Markov Chains are great for generating text based on previously seen text. 
        Here we'll either read from file or from one big string, then generate a 
        probabilistic understanding of the document by using ngrams as keys and
        storing all possible following words. We can then generate sentences
        using random dice and this object.
        ---
        Inputs
            text: either the path to a file containing the text or the text (string)
            from_file: whether the text is in a file or note (bool)
            ngram: how many words to use as a key for the text generation
            random_state: used to set the random state for reproducibility
        """
        self.ngram = int(ngram)
        self.markov_keys = dict()
        self._from_file = from_file
        if type(text) != type("string"):
            raise TypeError("'text' must be a PATH or string object")
        if from_file:
            self.path = text
        else:
            self.raw = text
        self.text_as_list = None
        if random_state:
            np.random.seed(random_state)
        self.create_probability_object()

    def preprocess(self):
        """
        Opens and cleans the text to be learned. If self.from_file, it reads
        from the path provided. The cleaning is very minor, just lowercasing
        and getting rid of quotes. Creates a list of words from the text.
        """
        if self._from_file:
            with open(self.path,'r') as f:
                self.raw = f.read()
        self.text_as_list = self.raw.lower().replace('"','').replace("'","").split()

    def markov_group_generator(self,text_as_list):
        """
        Generator that creates the ngram groupings to act as keys.
        Just grabs ngram number of words and puts them into a tuple
        and yields that upon iteration request.
        ---
        Inputs
            text_as_list: the text after preprocessing (list)
        Outputs
            keys: word groupings of length self.ngram (tuple)
        """
        if len(text_as_list) < self.ngram+1:
            raise ValueError("NOT A LONG ENOUGH TEXT!")
            return

        for i in range(self.ngram,len(text_as_list)):
            yield tuple(text_as_list[i-self.ngram:i+1])

    def create_probability_object(self):
        """
        Steps through the text, pulling keys out and keeping track
        of which words follow the keys. Duplication is allowed for 
        values for each key - but all keys are unique.
        """
        if self.markov_keys:
            print("Probability Object already built!")
            return
        if not self.text_as_list:
            self.preprocess()
        for group in self.markov_group_generator(self.text_as_list):
            word_key = tuple(group[:-1])
            if word_key in self.markov_keys:
                self.markov_keys[word_key].append(group[-1])
            else:
                self.markov_keys[word_key] = [group[-1]]
    
    def generate_sentence(self, length=25, starting_word_id=None):
        """
        Given a seed word, pulls the key associated with that word and 
        samples from the values available. Then moves to the newly generated 
        word and gets the key associated with it, and generates again. 
        Repeats until the sentence is 'length' words long.
        ---
        Inputs
            length: how many words to generate (int)
            starting_word_id: what word to use as seed, by location (int)
        Outputs
            gen_words: the generated sentence, including seed words (string)
        """
        if not self.markov_keys:
            raise ValueError("No probability object built. Check initialization!")
        
        if (not starting_word_id or type(starting_word_id) != type(int(1)) 
            or starting_word_id < 0 or starting_word_id > len(self.text_as_list)-self.ngram):
            starting_word_id = np.random.randint(0,len(self.text_as_list)-self.ngram)
            
        gen_words = self.text_as_list[starting_word_id:starting_word_id+self.ngram]
        
        while len(gen_words) < length:
            seed = tuple(gen_words[-self.ngram:])
            gen_words.append(np.random.choice(self.markov_keys[seed]))
        return ' '.join(gen_words)
    
    def print_key_value_pairs(self, num_keys=20):
        """
        Iterates through the probability object, printing key-value
        pairs. 
        ---
        Input
        num_keys: how many pairs to show (int)
        """
        i = 1
        for key,value in self.markov_keys.items():
            print(key,value)
            print()
            i+=1
            if i>int(num_keys):
                break