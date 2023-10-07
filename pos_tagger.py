from multiprocessing import Pool
from constants import *
import numpy as np
import time
from utils import *
from itertools import tee

from typing import List

""" Contains the part of speech tagger class. """


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
        self.num_words = -1

    
    
    def get_unigrams(self):
        """
        Computes unigrams. 
        Tip. Map each tag to an integer and store the unigrams in a numpy array. 
        """
        unigrams = [sum(x.count(tag) for x in self.data_tags) / self.num_words for tag in self.all_tags]
        self.unigrams = np.array(unigrams)

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
                bigrams[i] += LAPLACE_FACTOR / len(self.all_tags) / (self.unigrams[i] * self.num_words + LAPLACE_FACTOR)
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
                lexical[self.tag2idx[tag], self.word2idx[word]] += 1 / len(self.all_tags)
        self.lexical = lexical 
    

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
        self.get_trigrams()


    def sequence_probability(self, sequence, tags):
        """Computes the probability of a tagged sequence given the emission/transition
        probabilities.
        """
        # TODO: change 
        if self.unigrams is None:
            self.get_unigrams()
        if self.lexical is None:
            self.get_emissions()
        # TODO: consider working in log space
        log_probability = 0
        for tag, word in zip(tags, sequence):
            # handle unknown words
            if word not in self.word2idx.keys():
                return 0
            log_probability += np.log(self.lexical[self.tag2idx[tag], self.word2idx[word]])
            log_probability += np.log(self.unigrams[self.tag2idx[tag]])
        return np.exp(log_probability)
        ## TODO

    def greedy(self, sequence):
        """Decodes the most likely sequence using greedy decoding."""
        if self.lexical is None:
            self.get_emissions()
        current_sentence = []
        result = []
        for word in sequence:
            if word not in self.word2idx:
                best_tag = 'NNP'
            else: 
                best_tag = self.idx2tag[np.argmax(self.lexical[:, self.word2idx[word]])]
            current_sentence.append(word)
            result.append(best_tag)
        return result

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
        ## TODO
        return []

if __name__ == "__main__":
    pos_tagger = POSTagger(GREEDY, smoothing_method=LAPLACE)

    train_data = load_data("data/train_x.csv", "data/train_y.csv")
    dev_data = load_data("data/dev_x.csv", "data/dev_y.csv")
    test_data = load_data("data/test_x.csv")

    pos_tagger.train(train_data)

    # Experiment with your decoder using greedy decoding, beam search, viterbi...

    # Here you can also implement experiments that compare different styles of decoding,
    # smoothing, n-grams, etc.
    evaluate(dev_data, pos_tagger)

    # Predict tags for the test set
    test_predictions = []
    for sentence in test_data:
        test_predictions.extend(pos_tagger.inference(sentence))
    
    # Write them to a file to update the leaderboard
    # TODO
