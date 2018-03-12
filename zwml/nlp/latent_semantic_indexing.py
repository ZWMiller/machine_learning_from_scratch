import numpy as np

class latent_semantic_indexing:
    
    def __init__(self, num_topics=5):
        """
        Latent semantic indexing uses matrix decomposition
        techniques to reduce the large feature space associated
        with text analysis into a smaller "topic" space which
        by exploiting SVD's ability to find correlations in
        features and combine them into super-dimensions made
        of the correlated columns. In the text analysis, that 
        means if the original features are word, LSI will 
        find words that tend to be in the same document together
        and group them as unique topics. 
        """
        self.num_topics = num_topics
        
    def fit(self, X):
        """
        Using SVD as the base of the algorithm (we use numpy since 
        it's faster than our method), we do a dimensionality
        reduction. Remember that V is an expression of the new
        dimensions in terms of the old columns. If we do count
        vectorizer, this is an expression of topics in terms of
        ngrams. We'll use this to extract our topics. We can also
        cast new documents into topic space using the V matrix.
        """
        X = self.convert_to_array(X)
        self.U, self.sigma, self.V = np.linalg.svd(X)
        self.V = self.V[:self.num_topics,:]
        self.sigma = self.sigma[:self.num_topics]
        self.U = self.U[:,:self.num_topics]
        
    def transform(self, X):
        """
        Since V is a conversion of columns to the lower
        dimensional space, we can just use matrix 
        multiplication to cast any new data into that 
        space.
        ---
        Input: X, data matrix (dataframe, array, list of lists)
        """
        X = self.convert_to_array(X)
        return np.dot(X, self.V.T)
    
    def fit_transform(self, X):
        """
        Fit on X and then transform X and return it as vectors.
        """
        self.fit(X)
        return self.transform(X)
    
    def print_topics(self, X, id_to_word=None, num_words_per_topics=10):
        """
        For each topic created in the SVD decomposition,
        iterate through the strongest contributors (positive
        or negative), and print out those words. Requires a 
        column number to word dictionary, otherwise just prints
        the column number for the strong correlations.
        """
        for idx, row in enumerate(self.V):
            sorted_word_ids = np.argsort(row)[-num_words_per_topics:]
            print("--- Topic ", idx, " ---")
            words_to_print = ""
            for word_id in sorted_word_ids:
                if id_to_word != None:
                    words_to_print += id_to_word[word_id]
                    words_to_print += ', '
                else:
                    words_to_print += "Column "
                    words_to_print += str(word_id)
                    words_to_print += ', '
            print(words_to_print[:-2])
    
    def pandas_to_numpy(self, x):
        """
        Checks if the input is a Dataframe or series, converts to numpy matrix for
        calculation purposes.
        ---
        Input: X (array, dataframe, or series)
        Output: X (array)
        """
        if type(x) == type(pd.DataFrame()) or type(x) == type(pd.Series()):
            return x.as_matrix()
        if type(x) == type(np.array([1,2])):
            return x
        return np.array(x) 
    
    def handle_1d_data(self,x):
        """
        Converts 1 dimensional data into a series of rows with 1 columns
        instead of 1 row with many columns.
        """
        if x.ndim == 1:
            x = x.reshape(-1,1)
        return x
    
    def convert_to_array(self, x):
        """
        Takes in an input and converts it to a numpy array
        and then checks if it needs to be reshaped for us
        to use it properly
        """
        x = self.pandas_to_numpy(x)
        x = self.handle_1d_data(x)
        return x
                        