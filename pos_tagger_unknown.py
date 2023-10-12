import gensim
import string
import time
from multiprocessing import Pool
import numpy as np
from gensim.models import Word2Vec
from sklearn.decomposition import PCA
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
from constants import *
from utils import *
import joblib
from sklearn.preprocessing import LabelEncoder, MinMaxScaler

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
    processes = 8
    sentences = data[0]
    words, tags = flatten_data(data[0],data[1])
    n = len(words)
    k = n//processes
    unk_n_tokens = sum([1 for s in sentences for w in s if w not in model.word2idx.keys()])
    # Identify and isolate sentences with unknown words and their respective tags
    words = [word for word in words if word not in model.word2idx.keys()]
    tags = [tag for word, tag in zip(words, tags) if word not in model.word2idx.keys()]
    predictions = {i:None for i in range(unk_n_tokens)}
         
    start = time.time()
    pool = Pool(processes=processes)
    res = []
    for i in range(0, n, k):
        res.append(pool.apply_async(infer_sentences, [model, words[i:i+k], i]))
    ans = [r.get(timeout=None) for r in res]
    predictions = dict()
    for a in ans:
        predictions.update(a)
    print(f"Inference Runtime: {(time.time()-start)/60} minutes.")
    
    unk_token_acc = sum([1 for i in range(len(words)) if tags[i] == predictions[i] and words[i] not in model.word2idx.keys()]) / unk_n_tokens
    #adding my manual code
    misclassification_count = 0
    for i in range(len(words)):
        # Check if the word is unknown and misclassified
        if words[i] not in model.word2idx.keys() and tags[i] != predictions[i]:
            print(f"Sentence: {i}, Word: {words[i]}, Actual Tag: {tags[i]}, Predicted Tag: {predictions[i]}")
            misclassification_count += 1
        # Stop after printing 10 misclassifications
        if misclassification_count >= 20:
            break
    print("Unk token acc: {}".format(unk_token_acc))
    return

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
    def __init__(self,documents,model_type="MLP"):
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
        self.build_prefix_indices(documents)
        if model_type == "XGB":
            self.label_encoder = LabelEncoder()
        self.model_type = model_type
        self.scaler = MinMaxScaler(feature_range=(-1, 1))
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
        # Unigram suffixes
        self.uni_suffixes = list(set([word[-1:] for sentence in documents for word in sentence]))
        self.uni_suffixes.append('...')
        self.uni_suffix2idx = {self.uni_suffixes[i]:i for i in range(len(self.uni_suffixes))}
        self.idx2uni_suffix = {v:k for k,v in self.uni_suffix2idx.items()}

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
    
    def get_uni_suffix_index(self,word):
        try:
            return self.uni_suffix2idx[word[-1:]]
        except KeyError:
            return self.uni_suffix2idx['...']
        
    def build_prefix_indices(self, documents):
        """Extract prefixes from words and map them to indices."""
        self.tri_prefixes = list(set([word[:3] for sentence in documents for word in sentence]))
        self.tri_prefixes.append('...')
        self.tri_prefix2idx = {self.tri_prefixes[i]: i for i in range(len(self.tri_prefixes))}
        self.idx2tri_prefix = {v: k for k, v in self.tri_prefix2idx.items()}

        self.bi_prefixes = list(set([word[:2] for sentence in documents for word in sentence]))
        self.bi_prefixes.append('...')
        self.bi_prefix2idx = {self.bi_prefixes[i]: i for i in range(len(self.bi_prefixes))}
        self.idx2bi_prefix = {v: k for k, v in self.bi_prefix2idx.items()}

    def get_tri_prefix_index(self, word):
        try:
            return self.tri_prefix2idx[word[:3]]
        except KeyError:
            return self.tri_prefix2idx['...']

    def get_bi_prefix_index(self, word):
        try:
            return self.bi_prefix2idx[word[:2]]
        except KeyError:
            return self.bi_prefix2idx['...']

    def predict_word(self,word):
        vector_word = self.get_word_vector(word)
        vector_word = self.scaler.fit_transform([vector_word])[0]
        if self.model_type == "MLP":
            return self.clf.predict([vector_word])[0]
        else:
            # Decode numerical prediction back to original label
            prediction = self.clf.predict([vector_word])[0]
            decoded_prediction = self.label_encoder.inverse_transform([prediction])[0]
            return decoded_prediction
    
    def inference(self,word):
        vector_word = self.get_word_vector(word)
        vector_word = self.scaler.fit_transform([vector_word])[0]
        if self.model_type == "MLP":
            return self.clf.predict([vector_word])[0]
        else:
            # Decode numerical prediction back to original label
            prediction = self.clf.predict([vector_word])[0]
            decoded_prediction = self.label_encoder.inverse_transform([prediction])[0]
            return decoded_prediction
    
    def get_w2v_embedding(self, word, vector_length=300):
        try:
            return self.model[word]
        except KeyError:
            return np.zeros(vector_length)
            
    def get_word_vector(self, word):
        features = {
            'tri_suffix_index': self.get_tri_suffix_index(word),
            'bi_suffix_index': self.get_bi_suffix_index(word),
            'uni_suffix_index': self.get_uni_suffix_index(word),
            'is_capitalized': int(word[0].isupper()),
            'num_capitals': sum([1 for char in word if char.isupper()]),
            'contains_dash': int('-' in word),
            'word_length': len(word),
            'contains_slash': int('\/' in word),
            'contains_number': int(any(char.isdigit() for char in word)),
            # Add other features as needed
        }
        try:
            full_embedding = self.model[word]
            reduced_embedding = self.pca.transform([full_embedding])[0]
             # Combine features: [reduced_embedding, suffix_index, is_capitalized, contains_dash, contains_number]
            feature_vector = np.concatenate(([features['tri_suffix_index'],features['bi_suffix_index'],
                                              features['uni_suffix_index'],features['num_capitals'],features['word_length'],
                                              features['contains_slash'],
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
                                                  features['uni_suffix_index'],features['num_capitals'],features['word_length'],
                                              features['contains_slash'],
                                            features['is_capitalized'], features['contains_dash'],
                                            features['contains_number']], reduced_embedding))
                return feature_vector
            else:
                reduced_embedding = np.zeros(self.embeddings_new_length)
                feature_vector = np.concatenate(([features['tri_suffix_index'],features['bi_suffix_index'],
                                                  features['uni_suffix_index'],features['num_capitals'],features['word_length'],
                                              features['contains_slash'],
                                            features['is_capitalized'], features['contains_dash'],
                                            features['contains_number']], reduced_embedding))
                return feature_vector
    
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
        filtered_words = self.scaler.fit_transform(filtered_words)
        if self.model_type == "MLP":
            self.clf = MLPClassifier(hidden_layer_sizes=(100,100,50),max_iter=300)
            self.clf.fit(np.array(filtered_words), np.array(filtered_labels))
        else:
            # Utilize XGBoost
            encoded_labels = self.label_encoder.fit_transform(train_y)
            self.clf = xgb.XGBClassifier(objective='multi:softprob',  # Softmax probability for multi-class classification
                num_class=len(set(encoded_labels))
                )
            self.clf.fit(np.array(filtered_words), np.array(encoded_labels))
        print("finished training unknown word POS tagger...")
        #joblib.dump(self.clf, './unknown_tagger.joblib')

    def load_model(self,model_path):
        self.clf = joblib.load(model_path)

if __name__ == "__main__":
    """Avoid using this script for testing as it's currently broken at the moment."""
    train_data = load_data("data/train_x.csv", "data/train_y.csv")
    train_x, train_y = flatten_data(train_data[0],train_data[1])
    dev_data = load_data("data/dev_x.csv", "data/dev_y.csv")
    test_data = load_data("data/test_x.csv")
    clf = POSTagger_MLP(train_data[0], model_type="XBG")
    clf.train(np.array(train_x[:2000]),np.array(train_y[:2000]))
    evaluate(dev_data[:1000],clf)