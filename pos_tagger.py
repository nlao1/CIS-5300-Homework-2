from multiprocessing import Pool
from constants import *
import numpy as np
import time
import string
from utils import *
from itertools import tee
import gensim
from gensim.models import Word2Vec
from gensim.models import FastText
from typing import List
from sklearn import svm
from sklearn.decomposition import PCA
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
from gensim.models.fasttext import load_facebook_vectors
import xgboost as xgb
import csv

""" Contains the part of speech tagger class. """

def flatten_data(data_words, data_tags):
    flattened_words = [word for sentence in data_words for word in sentence]
    flattened_tags = [tag for tag_list in data_tags for tag in tag_list]
    return flattened_words, flattened_tags

def has_numbers(inputString):
    return any(char.isdigit() for char in inputString)

def contains_dash(inputString):
    return '-' in inputString

def is_punctuation(word):
    return word in string.punctuation

class POSTagger_MLP():
    def __init__(self,documents):
        """Initializes the tagger model parameters and anything else necessary. """
        # Train Word2Vec model
        model_path = "./GoogleNews-vectors-negative300.bin"
        self.model = gensim.models.KeyedVectors.load_word2vec_format(model_path, binary=True)
        # Fit PCA on training data once here
        all_embeddings = [self.model[word] for sentence in documents for word in sentence if word in self.model]
        self.embeddings_new_length = 100
        self.pca = PCA(n_components=self.embeddings_new_length).fit(all_embeddings)
        #Create set of tokens
        self.build_vocab(documents)
        #Create freq
        freq = [sum(doc.count(word) for doc in documents) for word in self.vocab]
        self.freq = np.array(freq)
        #suffix vocab
        self.build_suffix_indices(documents)
        print("initialized unknown word POS tagger...")

    def build_vocab(self, documents):
        """Build a vocabulary and map words to indices. Filter words that occur more than 10 times."""
        self.vocab = list(set([word for sentence in documents for word in sentence]))
        self.word2idx = {self.vocab[i]:i for i in range(len(self.vocab))}
        self.idx2word = {v:k for k,v in self.word2idx.items()}

    def build_suffix_indices(self, documents):
        """Extract suffixes from words and map them to indices."""
        # Implement suffix extraction logic
        self.tri_suffixes = list(set([word[-3:] for sentence in documents for word in sentence]))
        self.tri_suffixes.append('...')
        self.tri_suffix2idx = {self.tri_suffixes[i]:i for i in range(len(self.tri_suffixes))}
        self.idx2tri_suffix = {v:k for k,v in self.tri_suffix2idx.items()}
        #do it for last two characters
        self.bi_suffixes = list(set([word[-2:] for sentence in documents for word in sentence]))
        self.bi_suffixes.append('...')
        self.bi_suffix2idx = {self.bi_suffixes[i]:i for i in range(len(self.bi_suffixes))}
        self.idx2bi_suffix = {v:k for k,v in self.bi_suffix2idx.items()}

    def get_tri_suffix_index(self, word):
        try:
            return self.tri_suffix2idx[word[-3:]]
        except KeyError:
            return self.tri_suffix2idx['...']
    def get_bi_suffix_index(self,word):
        try:
            return self.bi_suffix2idx[word[-2:]]
        except KeyError:
            return self.tri_suffix2idx['...']
        
    def predict_words(self,words):
        vector_words = self.get_word_vectors(words)
        return self.clf.predict(vector_words)

    def predict_word(self,word):
        vector_word = self.get_word_vector(word)
        return self.clf.predict([vector_word])[0]
    
    def get_w2v_embedding(self, word, vector_length=300):
        try:
            return self.model[word]
        except KeyError:
            return np.zeros(vector_length)
            
    def get_word_vector(self, word):
        features = {
            'tri_suffix_index': self.get_tri_suffix_index(word),
            'bi_suffix_index': self.get_bi_suffix_index(word),
            'is_capitalized': int(word[0].isupper()),
            'contains_dash': int('-' in word),
            'contains_number': int(any(char.isdigit() for char in word)),
            # Add other features as needed
        }
        try:
            full_embedding = self.model[word]
            reduced_embedding = self.pca.transform([full_embedding])[0]
             # Combine features: [reduced_embedding, suffix_index, is_capitalized, contains_dash, contains_number]
            feature_vector = np.concatenate(([features['tri_suffix_index'],features['bi_suffix_index'],
                                            features['is_capitalized'], features['contains_dash'],
                                            features['contains_number']], reduced_embedding))
            
            return feature_vector
        except KeyError:
            # Handle the case where the word is not in the vocabulary
            #print(f"{word} not found in word2vec")
            if contains_dash(word):
                temp = word.split('-')
                vectors = []
                for word in temp:
                    vectors.append(self.get_w2v_embedding(word))
                full_embedding = np.sum(vectors,axis=0)/len(vectors)
                reduced_embedding = self.pca.transform([full_embedding])[0]
                feature_vector = np.concatenate(([features['tri_suffix_index'],features['bi_suffix_index'],
                                            features['is_capitalized'], features['contains_dash'],
                                            features['contains_number']], reduced_embedding))
                return feature_vector
            else:
                reduced_embedding = np.zeros(self.embeddings_new_length)
                feature_vector = np.concatenate(([features['tri_suffix_index'],features['bi_suffix_index'],
                                            features['is_capitalized'], features['contains_dash'],
                                            features['contains_number']], reduced_embedding))
                return feature_vector

    def get_word_vectors(self, words):
        vectors = []
        for word in words:
            vectors.append(self.get_word_vector(word))
        return vectors
    
    #assume np arrays
    def train(self,train_x,train_y):
        print("training unknown word POS tagger...")
        #get the words & tag labels and create feature vectors with word2vec
        filtered_words = []
        filtered_labels = []
        for word, label in zip(train_x, train_y):
            if self.freq[self.word2idx[word]] >= 10:
                #print("ah")
                pass
            else:
                vector = self.get_word_vector(word)
                # if np.any(vector):  # Check if the vector is non-zero
                filtered_words.append(vector)  # Store the vector instead of the word
                filtered_labels.append(label)
        self.clf = MLPClassifier(hidden_layer_sizes=(100,100,50),max_iter=300)
        # Utilize XGBoost
        #self.clf = xgb.XGBClassifier(objective='multi:softprob',  # Softmax probability for multi-class classification
        #    num_class=len(set(train_y))
         #   )
        self.clf.fit(np.array(filtered_words), np.array(filtered_labels))
        #self.clf = RandomForestClassifier()
        print("finished training unknown word POS tagger...")

def evaluate(data, model):
    """Evaluates the POS model on some sentences and gold tags.

    This model can compute a few different accuracies:
        - whole-sentence accuracy
        - per-token accuracy
        - compare the probabilities computed by different styles of decoding

    You might want to refactor this into several different evaluation functions,
    or you can use it as is. 
    
    As per the write-up, you may find it faster to use multiprocessing (code included). 
    
    """
    processes = 4
    sentences = data[0]
    tags = data[1]
    n = len(sentences)
    k = n//processes
    n_tokens = sum([len(d) for d in sentences])
    unk_n_tokens = sum([1 for s in sentences for w in s if w not in model.word2idx.keys()])
    predictions = {i:None for i in range(n)}
    probabilities = {i:None for i in range(n)}
         
    start = time.time()
    pool = Pool(processes=processes)
    res = []
    for i in range(0, n, k):
        res.append(pool.apply_async(infer_sentences, [model, sentences[i:i+k], i]))
    ans = [r.get(timeout=None) for r in res]
    predictions = dict()
    for a in ans:
        predictions.update(a)
    print(f"Inference Runtime: {(time.time()-start)/60} minutes.")
    
    start = time.time()
    pool = Pool(processes=processes)
    res = []
    for i in range(0, n, k):
        res.append(pool.apply_async(compute_prob, [model, sentences[i:i+k], tags[i:i+k], i]))
    ans = [r.get(timeout=None) for r in res]
    probabilities = dict()
    for a in ans:
        probabilities.update(a)
    print(f"Probability Estimation Runtime: {(time.time()-start)/60} minutes.")


    token_acc = sum([1 for i in range(n) for j in range(len(sentences[i])) if tags[i][j] == predictions[i][j]]) / n_tokens
    unk_token_acc = sum([1 for i in range(n) for j in range(len(sentences[i])) if tags[i][j] == predictions[i][j] and sentences[i][j] not in model.word2idx.keys()]) / unk_n_tokens
    #adding my manual code
    misclassification_count = 0
    for i in range(n):
        for j in range(len(sentences[i])):
            # Check if the word is unknown and misclassified
            if sentences[i][j] not in model.word2idx.keys() and tags[i][j] != predictions[i][j]:
                print(f"Sentence: {i}, Word: {sentences[i][j]}, Actual Tag: {tags[i][j]}, Predicted Tag: {predictions[i][j]}")
                misclassification_count += 1
                # Stop after printing 10 misclassifications
                if misclassification_count >= 10:
                    break
        if misclassification_count >= 10:
            break
    print("now look at words that are known but still predicted incorrecly.")
    misclassification_count = 0 
    for i in range(n):
        for j in range(len(sentences[i])):
            # Check if the word is unknown and misclassified
            if sentences[i][j] in model.word2idx.keys() and tags[i][j] != predictions[i][j]:
                print(f"Sentence: {i}, Word: {sentences[i][j]}, Actual Tag: {tags[i][j]}, Predicted Tag: {predictions[i][j]}")
                misclassification_count += 1
                # Stop after printing 10 misclassifications
                if misclassification_count >= 10:
                    break
        if misclassification_count >= 10:
            break
    #end of manual code
    whole_sent_acc = 0
    num_whole_sent = 0
    for k in range(n):
        sent = sentences[k]
        eos_idxes = indices(sent, '.')
        start_idx = 1
        end_idx = eos_idxes[0]
        for i in range(1, len(eos_idxes)):
            whole_sent_acc += 1 if tags[k][start_idx:end_idx] == predictions[k][start_idx:end_idx] else 0
            num_whole_sent += 1
            start_idx = end_idx+1
            end_idx = eos_idxes[i]
    print("Whole sent acc: {}".format(whole_sent_acc/num_whole_sent))
    print("Mean Probabilities: {}".format(sum(probabilities.values())/n))
    print("Token acc: {}".format(token_acc))
    print("Unk token acc: {}".format(unk_token_acc))
    
    confusion_matrix(pos_tagger.tag2idx, pos_tagger.idx2tag, predictions.values(), tags, 'cm.png')

    return whole_sent_acc/num_whole_sent, token_acc, sum(probabilities.values())/n


class POSTagger():
    def __init__(self, inference_method, smoothing_method=None):
        """Initializes the tagger model parameters and anything else necessary. """
        self.smoothing_method = smoothing_method
        self.inference_method = inference_method
        self.unigrams = None
        self.bigrams = None
        self.trigrams = None
        self.lexical = None
        self.ngram = None
        self.num_words = -1

    def get_unigrams(self):
        """
        Computes unigrams. 
        Tip. Map each tag to an integer and store the unigrams in a numpy array. 
        """
        unigrams = [sum(x.count(tag) for x in self.data_tags) / self.num_words for tag in self.all_tags]
        self.unigrams = np.array(unigrams)

    def get_unigrams_words(self):
        """
        Computes unigrams. 
        Tip. Map each tag to an integer and store the unigrams in a numpy array. 
        """
        unigrams = [sum(x.count(word) for x in self.data_words) / self.num_words for word in self.all_words]
        self.unigrams_words = np.array(unigrams)

    def get_bigrams(self):        
        """
        Computes bigrams. 
        Tip. Map each tag to an integer and store the bigrams in a numpy array
             such that bigrams[index[tag1], index[tag2]] = Prob(tag2|tag1). 
        """
        if self.unigrams is None:
           self.get_unigrams()
        bigrams = np.zeros((len(self.all_tags), len(self.all_tags)))
        def pairwise(iterable):
            "s -> (s0,s1), (s1,s2), (s2, s3), ..."
            a, b = tee(iterable)
            next(b, None)
            return zip(a, b) 
        for document in self.data_tags:
            for curr, next_word in pairwise(document):
                if self.smoothing_method == LAPLACE:
                    bigrams[self.tag2idx[curr], self.tag2idx[next_word]] += 1 / (LAPLACE_FACTOR + self.unigrams[self.tag2idx[curr]] * self.num_words)
                else: 
                    bigrams[self.tag2idx[curr], self.tag2idx[next_word]] += 1 / (self.unigrams[self.tag2idx[curr]] * self.num_words)
        if self.smoothing_method == LAPLACE:
            for i in range(len(bigrams)):
                num_log = np.log(LAPLACE_FACTOR) - np.log(len(self.all_tags))
                denom_log = np.log(self.unigrams[i] * self.num_words + LAPLACE_FACTOR)
                bigrams[i] += np.exp(num_log - denom_log)
        elif self.smoothing_method == INTERPOLATION:
            lambda_1, lambda_2 = BIGRAM_LAMBDAS
            for i in range(len(bigrams)):
                for j in range(len(bigrams[i])):
                    bigrams[i,j] = lambda_1 * bigrams[i,j] + lambda_2 * self.unigrams[j]
        self.bigrams = bigrams
    
    def get_trigrams(self):
        """
        Computes trigrams. 
        Tip. Similar logic to unigrams and bigrams. Store in numpy array. 
        """
        if self.bigrams is None:
           self.get_bigrams()
        trigrams = np.zeros((len(self.all_tags), len(self.all_tags), len(self.all_tags)))
        def triplewise(iterable):
            "s -> (s0,s1,s2), (s1,s2,s3), (s2, s3, s4), ..."
            a, b, c = tee(iterable, 3)
            next(b, None)
            next(c, None)
            next(c, None)
            return zip(a, b, c) 
        for document in self.data_tags:
            for curr, next_word, nextnext_word in triplewise(document):
                if self.smoothing_method == LAPLACE:
                    trigrams[self.tag2idx[curr], self.tag2idx[next_word], self.tag2idx[nextnext_word]] += 1 / (LAPLACE_FACTOR + self.bigrams[self.tag2idx[curr], self.tag2idx[next_word]] * self.unigrams[self.tag2idx[curr]] * self.num_words)
                else: 
                    trigrams[self.tag2idx[curr], self.tag2idx[next_word], self.tag2idx[nextnext_word]] += 1 / (self.bigrams[self.tag2idx[curr], self.tag2idx[next_word]] * self.unigrams[self.tag2idx[curr]] * self.num_words)
        if self.smoothing_method == LAPLACE:
            trigrams += LAPLACE_FACTOR / self.num_words
        elif self.smoothing_method == INTERPOLATION:
            lambda_1, lambda_2, lambda_3 = TRIGRAM_LAMBDAS
            for i in range(len(trigrams)):
                for j in range(len(trigrams[i])):
                    for k in range(len(trigrams[i,j])):
                        trigrams[i,j,k] = lambda_1 * trigrams[i,j,k] + lambda_2 * self.bigrams[j,k] + lambda_3 * self.unigrams[k]
        self.trigrams = trigrams
    
    
    def get_emissions(self):
        """
        Computes emission probabilities. 
        Tip. Map each tag to an integer and each word in the vocabulary to an integer. 
             Then create a numpy array such that lexical[index(tag), index(word)] = Prob(word|tag) 
        """
        lexical = np.zeros((len(self.all_tags), len(self.all_words)))
        for document_words, document_tags in zip(self.data_words, self.data_tags):
            for word, tag in zip(document_words, document_tags):
                lexical[self.tag2idx[tag], self.word2idx[word]] += 1
        # Apply unigram prior smoothing and convert to probabilities.
        tag_counts = np.sum(lexical, axis=1, keepdims=True)
        
        # Ensure self.unigrams is a numpy array for efficient operations
        if not isinstance(self.unigrams, np.ndarray):
            self.unigrams = np.array(self.unigrams)

        # Ensure unigrams are normalized to represent probabilities
        unigram_probs = self.unigrams_words / np.sum(self.unigrams_words)

        # Apply the smoothing
        temp = None
        if self.smoothing_method == LAPLACE:
            temp = np.log(LAPLACE_FACTOR) +  np.log(1) - np.log(len(self.all_words))
        else:
            temp = np.log(LAPLACE_FACTOR) + np.log(unigram_probs)
        lexical = np.log(lexical + np.exp(temp)) - np.log(tag_counts + LAPLACE_FACTOR)
        # Convert to log-probabilities to avoid underflow and store.
        self.lexical = np.exp(lexical)

        #old code
        #lexical = np.log(lexical) - np.log(LAPLACE_FACTOR + self.num_words)
        #lexical = lexical - np.log(self.unigrams.reshape(-1,1))
        #self.lexical = np.exp(lexical)

    

    def train(self, data, ngram=2):
        """Trains the model by computing transition and emission probabilities.

        You should also experiment:
            - smoothing.
            - N-gram models with varying N.
        
        """
        self.data = data
        self.data_words : List[List[str]] = data[0]
        self.data_tags : List[List[str]] = data[1]
        self.all_tags = list(set([t for tag in data[1] for t in tag]))
        self.tag2idx = {self.all_tags[i]:i for i in range(len(self.all_tags))}
        self.idx2tag = {v:k for k,v in self.tag2idx.items()}

        self.all_words = list(set([word for sentence in self.data_words for word in sentence]))
        self.word2idx = {self.all_words[i]:i for i in range(len(self.all_words))}
        self.idx2word = {v:k for k,v in self.word2idx.items()}
        self.num_words = sum(len(d) for d in self.data_words)
        self.ngram = ngram
        self.clf = POSTagger_MLP(data[0])
        train_x, train_y = flatten_data(data[0],data[1])
        self.clf.train(np.array(train_x),np.array(train_y))
        self.get_trigrams()
        self.get_unigrams_words()

    def sequence_probability(self, sequence, tags):
        """Computes the probability of a tagged sequence given the emission/transition
        probabilities.
        """
        # TODO: change 
        if self.trigrams is None:
            self.get_trigrams()
        if self.lexical is None:
            self.get_emissions()
        log_probability = 0
        prev_prev_tag = None
        prev_tag = None
        for tag, word in zip(tags, sequence):
            # handle unknown words
            if word not in self.word2idx.keys():
                return 0
            log_probability += np.log(self.lexical[self.tag2idx[tag], self.word2idx[word]])
            if self.ngram == 1:
                log_probability += np.log(self.unigrams[self.tag2idx[tag]])
            elif self.ngram == 2:
                if prev_tag == None:
                    pass
                else:
                    log_probability += np.log(self.bigrams[self.tag2idx[prev_tag],self.tag2idx[tag]])
            else: 
                if prev_prev_tag == None:
                    if prev_tag == None:
                        pass
                    else:
                        log_probability += np.log(self.bigrams[self.tag2idx[prev_tag],self.tag2idx[tag]])
                else:
                    log_probability += np.log(self.trigrams[self.tag2idx[prev_prev_tag],self.tag2idx[prev_tag],self.tag2idx[tag]])
            prev_prev_tag = prev_tag
            prev_tag = tag
        return np.exp(log_probability)
    
    def get_tag_of_unknown(self, word):
        return self.clf.predict(word)

    def get_greedy_best_tag(self, word, prev_tag, prev_prev_tag):
        best_tag = None
        if self.ngram == 1:
            best_tag = self.idx2tag[np.argmax(self.lexical[:, self.word2idx[word]] * self.unigrams)]
        elif self.ngram == 2:
            if prev_tag is None:
                best_tag = 'O'
            else:
                best_tag_index = np.argmax(self.lexical[:, self.word2idx[word]] * self.bigrams[self.tag2idx[prev_tag], :])
                best_tag = self.idx2tag[best_tag_index]
            prev_tag = best_tag
        elif self.ngram == 3:
            if prev_tag is None: 
                best_tag = 'O'
            elif prev_prev_tag is None:
                best_tag_index = np.argmax(self.lexical[:, self.word2idx[word]] * self.bigrams[self.tag2idx[prev_tag], :]) 
                best_tag = self.idx2tag[best_tag_index]
            else: 
                best_tag_index = np.argmax(self.lexical[:, self.word2idx[word]] * self.trigrams[self.tag2idx[prev_prev_tag], self.tag2idx[prev_tag], :])
                best_tag = self.idx2tag[best_tag_index]
            prev_prev_tag = prev_tag
            prev_tag = best_tag
        return best_tag, prev_tag, prev_prev_tag

    def greedy(self, sequence):
        """Decodes the most likely sequence using greedy decoding."""
        if self.lexical is None:
            self.get_emissions()
        if self.trigrams is None:
            self.get_trigrams()
        prev_prev_tag = None
        prev_tag = None
        result = []
        for i, word in enumerate(sequence):
            best_tag = None
            if word not in self.word2idx:
                best_tag = self.clf.predict_word(word)
            else: 
                best_tag, prev_tag, prev_prev_tag = self.get_greedy_best_tag(word, prev_tag, prev_prev_tag)
            result.append(best_tag)
        return result
    
    def get_beam_search_best_tag(self, word, prev_tag, prev_prev_tag):
        best_tags = []
        if self.ngram == 1:
            best_tag_indices = np.argsort(self.lexical[:, self.word2idx[word]] * self.bigrams[self.tag2idx[prev_tag], :])[-BEAM_K:]
            best_tags = [self.idx2tag[index] for index in best_tag_indices]
        elif self.ngram == 2:
            if prev_tag is None:
                for i in range(BEAM_K):
                    best_tags.append('O')
            else:
                best_tag_indices = np.argsort(self.lexical[:, self.word2idx[word]] * self.bigrams[self.tag2idx[prev_tag], :])[-BEAM_K:]
                best_tags = [self.idx2tag[index] for index in best_tag_indices]
        elif self.ngram == 3:
            if prev_tag is None: 
                for i in range(BEAM_K):
                    best_tags.append('O')
            elif prev_prev_tag is None:
                best_tag_indices = np.argsort(self.lexical[:, self.word2idx[word]] * self.bigrams[self.tag2idx[prev_tag], :])[-BEAM_K:]
                best_tags = [self.idx2tag[index] for index in best_tag_indices]
            else: 
                best_tag_indices = np.argsort(self.lexical[:, self.word2idx[word]] * self.trigrams[self.tag2idx[prev_prev_tag], self.tag2idx[prev_tag], :])[-BEAM_K:]
                best_tags = [self.idx2tag[index] for index in best_tag_indices]
        return best_tags
    
    def beam_search(self, sequence):
        """Decodes the most likely sequence using beam-search or top-k greedy search."""
        if self.lexical is None:
            self.get_emissions()
        if self.trigrams is None:
            self.get_trigrams()     
        k_results = []
        for i in range(BEAM_K):
            k_results.append([])
        k_square_temp = []
        #we go through each word in the sequence
        for i, word in enumerate(sequence):
            k_tags = None
            #If word doesn't exist in dict, add arbitrary tag to each of the top-k decoded sequences 
            if word not in self.word2idx:
                for i in range(BEAM_K):
                    k_results[i].append('NNP')
            #Otherwise, find the best k tags for each of the top-k decoded sequences
            else:
                #go through each top decoded sequence
                for i in range(BEAM_K):
                    prev_prev_tag = None
                    prev_tag = None
                    if len(k_results[0]) >= 1:
                        prev_tag = k_results[i][-1]
                    if len(k_results[0]) >= 2:
                        prev_prev_tag = k_results[i][-2]
                    best_k_tags = self.get_beam_search_best_tag(word, prev_tag, prev_prev_tag)   
                    for tag in best_k_tags:
                        k_square_temp.append(k_results[i] + [tag])
                k_square_temp_with_pr = list(map(lambda x: (x, self.sequence_probability(x,sequence[:i+1])), k_square_temp))
                sorted(k_square_temp_with_pr, key= lambda x: x[1])
                k_results = k_square_temp_with_pr[-BEAM_K:]  
                k_results = list(map(lambda x: x[0], k_results))
                k_square_temp = []
        return k_results[-1]
    
    def viterbi_bigram(self, sequence):
        result = [None for _ in range(len(sequence))]
        pis = np.full((len(self.all_tags), len(sequence)), -np.inf)
        bps = np.full((len(self.all_tags), len(sequence)), None)
        # initialize first column
        pis[self.tag2idx['O'],0] = 0
        for i in range(1, len(sequence)):
            if sequence[i] not in self.word2idx.keys():
                # handle unknown here
                #predict 'NNP'
                nnp_index = self.tag2idx['NNP']
                pis[nnp_index,i] = np.max((pis[:,i-1] + np.log(self.bigrams[:, nnp_index])))
                bps[nnp_index,i] = np.argmax((pis[:,i-1] + np.log(self.bigrams[:, nnp_index])))
                continue
            for tag in self.all_tags:
                tag_idx = self.tag2idx[tag]
                emission = self.lexical[tag_idx, self.word2idx[sequence[i]]]
                best_prev_tag = np.argmax(pis[:,i-1] + np.log(self.bigrams[:,tag_idx]))
                best_pi = np.max(pis[:,i-1] + np.log(self.bigrams[:,tag_idx]))
                pis[tag_idx, i] = np.log(emission) + best_pi
                bps[tag_idx, i] = best_prev_tag
        best_final_tag_idx = np.argmax(pis[:,len(sequence)-1])
        result[len(sequence)-1] = self.idx2tag[best_final_tag_idx]
        best_prev_tag_idx = int(bps[best_final_tag_idx, len(sequence)-1])
        for i in range(len(sequence)-2, 0, -1):
            result[i] = self.idx2tag[best_prev_tag_idx]
            best_prev_tag_idx = int(bps[best_prev_tag_idx,i])
        result[0] = self.idx2tag[best_prev_tag_idx]
        return result

   
    def viterbi_trigram(self, sequence):
        result = [None for _ in range(len(sequence))]
        pis = np.full((len(self.all_tags) * len(self.all_tags), len(sequence)), -np.inf)
        bps = np.full((len(self.all_tags) * len(self.all_tags), len(sequence)), None)
        # initialize first column
        pis[self.tag2idx['O'],0] = 0
        for i in range(1, len(sequence)):
            if sequence[i] not in self.word2idx.keys():
                # handle unknown here
                #predict 'NNP'
                nnp_index = self.tag2idx['NNP']
                pis[nnp_index,i] = np.max((pis[:,i-1] + np.log(self.bigrams[:, nnp_index])))
                bps[nnp_index,i] = np.argmax((pis[:,i-1] + np.log(self.bigrams[:, nnp_index])))
                continue
            for tag in self.all_tags:
                tag_idx = self.tag2idx[tag]
                emission = self.lexical[tag_idx, self.word2idx[sequence[i]]]
                best_prev_tag = np.argmax(pis[:,i-1] + np.log(self.bigrams[:,tag_idx]))
                best_pi = np.max(pis[:,i-1] + np.log(self.bigrams[:,tag_idx]))
                pis[tag_idx, i] = np.log(emission) + best_pi
                bps[tag_idx, i] = best_prev_tag
        best_final_tag_idx = np.argmax(pis[:,len(sequence)-1])
        result[len(sequence)-1] = self.idx2tag[best_final_tag_idx]
        best_prev_tag_idx = int(bps[best_final_tag_idx, len(sequence)-1])
        for i in range(len(sequence)-2, 0, -1):
            result[i] = self.idx2tag[best_prev_tag_idx]
            best_prev_tag_idx = int(bps[best_prev_tag_idx,i])
        result[0] = self.idx2tag[best_prev_tag_idx]
        return result

    def viterbi(self, sequence):
        if self.lexical is None:
            self.get_emissions()
        if self.trigrams is None:
            self.get_trigrams()
        if self.ngram == 2: 
            return self.viterbi_bigram(sequence)
        elif self.ngram == 3:
            return self.viterbi_trigram(sequence)




    def inference(self, sequence):
        """Tags a sequence with part of speech tags.

        You should implement different kinds of inference (suggested as separate
        methods):

            - greedy decoding
            - decoding with beam search
            - viterbi
        """
        if self.inference_method == GREEDY:
            return self.greedy(sequence)
        elif self.inference_method == BEAM:
            return self.beam_search(sequence)
        elif self.inference_method == VITERBI:
            return self.viterbi(sequence)

if __name__ == "__main__":
    pos_tagger = POSTagger(GREEDY, smoothing_method=LAPLACE)

    train_data = load_data("data/train_x.csv", "data/train_y.csv")
    dev_data = load_data("data/dev_x.csv", "data/dev_y.csv")
    test_data = load_data("data/test_x.csv")

    pos_tagger.train(train_data, ngram=2)

    # Experiment with your decoder using greedy decoding, beam search, viterbi...

    # Here you can also implement experiments that compare different styles of decoding,
    # smoothing, n-grams, etc.
    evaluate(dev_data[:1000], pos_tagger)

    # Predict tags for the test set
    test_predictions = []
    for sentence in test_data:
        test_predictions.extend(pos_tagger.inference(sentence))
    
    # Write them to a file to update the leaderboard
    # Write them to a file to update the leaderboard
    with open('test_y.csv', mode='w', newline='') as file:
        writer = csv.writer(file)
        # Optionally write a header
        writer.writerow(["id","predicted_tag"])
        for i, tag in enumerate(test_predictions):
            writer.writerow([i,tag])
