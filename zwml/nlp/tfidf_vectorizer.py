import numpy as np
from collections import Counter
from string import punctuation

class tfidf_vectorizer:
    
    def __init__(self, max_features=None, ngrams = (1,1), tokenizer=None, remove_stopwords=False):
        """
        Term frequency, inverse document frequency vectorizer 
        reads the text provided, tokenizes it with the provided 
        tokenizer (or the default), then generates ngrams keeping 
        track of all ngrams as the vocabulary. Then it takes provided 
        texts and converts them into vectors by counting the 
        appearance of each ngram and tracking that for every document. 
        The counts are then scaled by the max term frequency and the
        inverse document frequency (see converter method). This new
        result is better than counts at picking out how important
        words are based on both usage and uniqueness. 
        ---
        KWargs:
        max_features: how many ngrams to allow in the vector, using the
        most common features first. If None, defaults to using all
        ngrams (int)
        ngrams: how many tokens to combine to form features. First element
        of tuple is starting point, second is ending point.
        tokenizer: what function to use to create tokens (must return 
        list of tokens)
        remove_stopwords: whether to include very common english words that
        do not add much value due to their commonness.
        """
        self.max_features = max_features
        self.vocabulary = {}
        self.ngrams = ngrams
        if tokenizer == None:
            self.tokenizer = self.tokenize
        else:
            self.tokenizer = tokenizer
        self.remove_stopwords = remove_stopwords
        self.stopwords = {'ourselves', 'hers', 'between', 'yourself', 'but', 'again', 
                          'there', 'about', 'once', 'during', 'out', 'very', 'having', 
                          'with', 'they', 'own', 'an', 'be', 'some', 'for', 'do', 'its', 
                          'yours', 'such', 'into', 'of', 'most', 'itself', 'other', 'off', 
                          'is', 's', 'am', 'or', 'who', 'as', 'from', 'him', 'each', 'the', 
                          'themselves', 'until', 'below', 'are', 'we', 'these', 'your', 
                          'his', 'through', 'don', 'nor', 'me', 'were', 'her', 'more', 
                          'himself', 'this', 'down', 'should', 'our', 'their', 'while', 
                          'above', 'both', 'up', 'to', 'ours', 'had', 'she', 'all', 'no', 
                          'when', 'at', 'any', 'before', 'them', 'same', 'and', 'been', 
                          'have', 'in', 'will', 'on', 'does', 'yourselves', 'then', 'that', 
                          'because', 'what', 'over', 'why', 'so', 'can', 'did', 'not', 'now', 
                          'under', 'he', 'you', 'herself', 'has', 'just', 'where', 'too', 'only', 
                          'myself', 'which', 'those', 'i', 'after', 'few', 'whom', 't', 'being', 
                          'if', 'theirs', 'my', 'against', 'a', 'by', 'doing', 'it', 'how', 'further', 
                          'was', 'here', 'than'}
        
    def token_generator(self, X):
        """
        Generator that returns joined tokens as a single
        string to act as a feature. It generates the tokens
        by iterating through the allowed ngrams and combining
        the appropriate number of tokens into a string.
        """
        for i in range(self.ngrams[0],self.ngrams[1]+1):
            for ix, _ in enumerate(X):
                if ix+i < len(X)+1:
                    yield ' '.join(X[ix:ix+i])
    
    def tokenize(self, X):
        """
        Simple tokenizer that removes punctuation,
        lowercases the text, and breaks on spaces.
        Also removes stopwords and numeric values
        from being treated as words.
        """
        for symbol in punctuation:
            X = X.replace(symbol,'')
        final_token_list = [] 
        for token in X.lower().split():
            if self.remove_stopwords:
                if not self.check_stopwords(token):
                    try:
                        int(token)
                        float(token)
                    except:
                        final_token_list.append(token)  
            else:
                final_token_list.append(token)
        return final_token_list
        
    def check_stopwords(self, token):
        """
        Checks if the token is in our list of common
        stopwords, and returns a boolean.
        """
        return token in self.stopwords
    
    def fit(self, X):
        """
        Go through all provided training documents and
        create the list of vocabulary for known documents
        by looking at all ngrams and tracking how often
        those ngrams appear. If max_features is defined,
        only keep the most common tokens. Afterward,
        generate a token_to_id mapper and an id_to_token
        mapper.
        """
        for document in X:
            tokens = self.tokenizer(document)
            for token in self.token_generator(tokens):
                if token in self.vocabulary.keys():
                    self.vocabulary[token] += 1
                else:
                    self.vocabulary[token] = 1
        
        if self.max_features != None:
            temp_vocab = {}
            for key, value in Counter(self.vocabulary).most_common(self.max_features):
                temp_vocab[key] = value
            self.vocabulary = temp_vocab
            del temp_vocab
            
        self.token_to_id = {ky: ix for ix, ky in enumerate(sorted(self.vocabulary.keys()))}
        self.id_to_token = {ix: ky for ix, ky in enumerate(sorted(self.vocabulary.keys()))}
        
        
    def transform(self, X):
        """
        Go through all provided documents and use the known
        vocabulary to track how often each ngram appears in
        the document. At the end, stack all of the generated
        document vectors together. Convert them to tf-idf
        and skip the initial vector that's all 0's, which 
        is just there to act as a template.
        """
        vectorized_docs = np.zeros(len(self.vocabulary.keys()))
        for document in X:
            tokens = self.tokenizer(document)
            vectorized_doc = np.zeros(len(self.vocabulary.keys()))
            for token in self.token_generator(tokens):
                if token in self.vocabulary:
                    word_id = self.token_to_id[token]
                    vectorized_doc[word_id] += 1
            vectorized_docs = np.vstack((vectorized_docs,vectorized_doc))
        return self.convert_counts_to_tf_idf(vectorized_docs)[1:]
    
    def convert_counts_to_tf_idf(self, docs):
        """
        To convert from counts to TF-IDF, we first scale
        each value by the maximum in it's own column. This 
        lowers dependence on document length. Then we calculate
        log(number of documents/(1+documents containing this ngram)).
        This is the inverse document frequency (the one is to make
        combat division by 0). Each value is scaled as:
        term_frequency*inverse_document_frequency.
        """
        number_of_columns = docs.shape[1]
        number_of_docs = docs.shape[0]
        frequency_scalers = np.ones(number_of_columns)
        idf_terms = np.ones(number_of_columns)
        for col in range(number_of_columns):
            column_vals = docs.T[col]
            frequency_scalers[col] = np.max(column_vals)
            number_of_docs_containing = np.sum((column_vals > 0).astype(int))
            idf_terms[col] = np.log(number_of_docs/(1+number_of_docs_containing))
        docs = docs/frequency_scalers
        docs = docs*idf_terms
        
        return docs           
    
    def fit_transform(self, X):
        """
        Fit on X and then transform X and return it as vectors.
        """
        self.fit(X)
        return self.transform(X)
                        