from __future__ import division
import argparse
import pandas as pd
import time
import json
import pickle
from tqdm import tqdm
# useful stuff
import numpy as np
from scipy.special import expit
import spacy
from sklearn.preprocessing import normalize
import os
import re

__authors__ = ['Louis Ternon', 'Pierre Sevestre', 'Marine Sobas', 'Vincent Ponslet']
__emails__ = ['louis.ternon@student.ecp.fr', 'pierre.sevestre@student.ecp.fr', 'marine.sobas@student.ecp.fr', 'vincent.ponslet@student.ecp.fr']


def raw_to_tokens(raw_string, spacy_nlp):
    string = re.sub('[^a-zA-Z ]+', "", raw_string.lower())
    spacy_tokens = spacy_nlp(string)
    string_tokens = [token.orth_ for token in spacy_tokens if not token.is_punct if not token.is_stop if
                     not token.is_currency if not token.like_num if not token.like_url if not token.orth_ in (' ', '')]  # if not token.is_currency]
    return string_tokens


def text2sentences(path, n=1e12):
    # feel free to make a better tokenization/pre-processing
    """
    :param path: path to the text files
    :return: list of all sentences
    """
    # TODO remove non-letter words

    sentences = []
    spacy_nlp = spacy.load('en')
    with open(path) as f:
        for q, l in tqdm(enumerate(f)):
            sentences.append(raw_to_tokens(l[:-1], spacy_nlp))
            if q == n-1:
                break
    return sentences

def similarity(word1, word2, w2id, W):

    """
    computes similiarity between the two words. unknown words are mapped to one common vector
    :param word1:
    :param word2:
    :return: a float \in [0,1] indicating the similarity (the higher the more similar)
    """
    try:
        a, b = int(w2id[word1]), int(w2id[word2])
        e1, e2 = W[a, :], W[b, :]
        return e1.T @ e2 / (np.linalg.norm(e1) * np.linalg.norm(e2))
    except:
        return 0



def loadPairs(path):

    """
    :param path: path to the csv of word pairs
    :return: zip object of word pairs
    """

    data = pd.read_csv(path, delimiter='\t')
    pairs = zip(data['word1'], data['word2'], data['similarity'])
    return pairs


class SkipGram():

    def __init__(self, sentences=1, nEmbed=30, negativeRate=0.0001, winSize=5, winNegSize=5, minCount=0, maxFreq=1,
                  monitor_results=False, experience_tag=""):

        self.valid_vocab, self.vocab_occ, self.eliminated_words = self.initialize_vocab(sentences, minCount, maxFreq)

        # set of valid words  over the whole dataset, dict of their occurences, and set of all eliminated words
        print(f"Num words : {len(self.valid_vocab)}; eliminated : {len(self.eliminated_words)}")
        self.w2id, self.id2w = self.word_id()  # word to ID mapping

        self.sentences = self.filter_sentences(self.eliminated_words, sentences)  # list of filtered sentences
        self.trainset = self.buildTrainSet(self.sentences, winSize)

        self.neg_vocab_set, self.n_neg = self.build_neg_vocab(self.valid_vocab, self.vocab_occ, negativeRate)
        self.neg_distrib = self.build_neg_distrib(self.neg_vocab_set, '3/4')

        self.nEmbed = nEmbed
        self.n_words = len(self.valid_vocab)  # number of words that appear at list five times
        self.winSize = winSize  # Window size
        self.winNegSize = winNegSize  # Number of negative words per target vector

        self.W = np.random.uniform(-0.5, 0.5, size=(self.n_words, self.nEmbed)) / self.nEmbed
        self.Wp = np.random.uniform(-0.5, 0.5, size=(self.n_words, self.nEmbed)) / self.nEmbed


        self.monitor_results = monitor_results
        self.output_path = os.path.join("output", experience_tag)
        if monitor_results:
            if not os.path.exists(self.output_path ):
                os.makedirs(self.output_path )
            else:
                print("{} already exists, might overwrite".format(self.output_path))

    def initialize_vocab(self, sentences, minCount, maxFreq):
        """
        Eliminates words that appear less than minCount times from the dataset
        Eliminates words that have a frequency greater than maxFreq. We chose this specificaly to remove over represented
        words like "the" or "he"
        :param sentences: list[list[str]]. List of all sentences
        :param minCount:
        :param maxFreq: in the initialisation of the class, it is set by default to 1 so no word is elimminated
        :return:
        """
        all_words = [word for sentence in sentences for word in sentence]  # list of all the words of all sentences
        all_words_unique = set(all_words)
        vocab_occ = {word: 0 for word in all_words_unique}
        for word in all_words:
            vocab_occ[word] += 1
        maxCount = len(all_words)*maxFreq
        valid_words_set1 = set(filter(lambda x: vocab_occ[x] > minCount, all_words_unique))
        valid_words_set2 = set(filter(lambda x: vocab_occ[x] < maxCount, all_words_unique))
        valid_words_set = valid_words_set1.intersection(valid_words_set2)
        invalid_words_set = all_words_unique - valid_words_set
        return valid_words_set, vocab_occ, invalid_words_set

    def word_id(self):

        """
        :param setntences: set of all words in the vocab, occ > minCount
        :return: dictionaries {word:id} and {id:word}
        """

        return {word: q for q, word in enumerate(self.valid_vocab)}, \
               {q: word for q, word in enumerate(self.valid_vocab)}

    def filter_sentences(self, invalid_vocab, sentences):
        """

        :param invalid_vocab: all words that must be extracted from the dataset
        :param sentences: list of sentences that constitute the dataset
        :return: sentences where all the invalid vocab has been deleted
        """
        for q, sentence in enumerate(sentences):
            to_del = []
            for p, word in enumerate(sentence):
                if word in invalid_vocab:
                    to_del.append(p)
            for p in to_del[::-1]:  # reverse index to be able to use del
                del sentences[q][p]

        return sentences

    def buildTrainSet(self, sentences, winSize, dynamic=False, shuffle=True):
        """

        :param sentences: list of filtered sentences
        :param winSize: context size
        :param dynamic: if True, builds dynamic context windows
        :param suffle: if True, all
        :return: 2-D array. Each row is center_word, context_word. If the context has k words, then there are k rows created
        """
        trainset = []
        for sentence in sentences:
            for k, word in enumerate(sentence):
                if dynamic == True:
                    K = np.random.randint(1, winSize + 1)
                else:
                    K = winSize
                contextBefore = sentence[k - K:k]
                contextAfter = sentence[k + 1:k + 1 + K]
                for contextWord in (contextBefore + contextAfter):
                    if contextWord != word:
                        trainset.append((word, contextWord))

        if shuffle:
            np.random.shuffle(trainset)

        return np.array(trainset)

    def build_neg_vocab(self, vocab, vocab_occ, neg_treshold):
        """

        :param vocab: all the word that will be in the train set and occur at least minCount times
        :param vocab_occ: dict where {valid word  : number of occurences}
        :param neg_treshold: maximum time a word can occur to be qualified for the negative words set
        :return neg_vocag_set: set of all the words that can be negatively sampled
        :return n_neg: len(neg_vocab)
        """

        n = np.sum(vocab_occ[word] for word in vocab)
        vocab_freq = {word: vocab_occ[word] / n for word in vocab}
        neg_vocab = list(filter(lambda x: vocab_freq[x] < neg_treshold, vocab))
        neg_vocab_set = set(neg_vocab)
        n_neg = len(neg_vocab)

        return neg_vocab_set, n_neg

    def build_neg_distrib(self, neg_vocab, distrib=None):

        """
        :param neg_vocab: words that are eligible to be negatively sampled
        :param distrib: type of distribution to the negative sampling
        :return: a dictionary with the sampling frequency of each word
        """

        if distrib == "basic":
            return self.build_neg_distrib_basic(neg_vocab)

        elif distrib == "3/4":
            return self.transform_34(neg_vocab)

    def build_neg_distrib_basic(self, neg_vocab):

        """
        :param neg_vocab: list of all the words that qualify for the negative vocabulary
        :return: a dictionary with the frequency of each word of the negative vocabulary
        """
        n = np.sum(self.vocab_occ[word] for word in neg_vocab)
        distr = {neg_w: self.vocab_occ[neg_w] / n for neg_w in neg_vocab}

        return distr

    def transform_34(self, neg_vocab):

        """
        :param neg_vocab: words that are eligible to be negatively sampled
        :return normed_distr: a dictionary with the frequency of each word of the negative vocabulary. Each word receives a ^0.75
        transformation. Then all frequencies are normalised.
        """

        n = np.sum(self.vocab_occ[word] for word in neg_vocab)
        distr = {neg_w: (self.vocab_occ[neg_w] / n) ** 0.75 for neg_w in neg_vocab}
        sum_ = sum(list(distr.values()))
        normed_distr = {k: v / sum_ for k, v in distr.items()}

        return normed_distr

    def parse_sentence_for_context(self, sentence, K):
        """
        the windSize is defined at the initialisation of the class

        :param K: int, size of the context window
        :param sentence: str,
        :return: list of center words list(str), and a list with their corresponding context (list(list(str))

        """
        center_words = []
        contexts = []
        for k, word in enumerate(sentence):
            # kp = np.random.randint(1, K+1) #dynamic window size
            context_before = sentence[k - K:k]
            context_after = sentence[k + 1:k + 1 + K]
            contexts.append(context_before + context_after)
            center_words.append(word)
        return center_words, contexts

    def select_negative_sampling(self, neg_distribution, K):
        """

        :param neg_vocab: set of the negative vocab
        :param neg_distribution: {neg_word : sample rate}
        :return: returns K negative words
        """

        return [np.random.choice(list(neg_distribution.keys()), p=list(neg_distribution.values())) for x in range(K)]

    def clippedSigm(self, x):
        return expit(np.clip(x, -6, 6))  # Clip values at 6 to avoid extreme values at the gradient

    def forward(self, centerBatch, contextBatch, negativeBatch, negWinSize):

        """
        :param self:
        :param center_word: word of the center context to evaluate
        :param target_context_word: one word of the context
        :param neg_words: list of negative words
        :return: loss function and intermediate gradients
        """

        gradW = np.zeros(self.W.shape)
        gradWp = np.zeros(self.Wp.shape)
        if self.monitor_results:
            loss = 0
            loss_pos = 0
            loss_neg = 0

        for q in range(len(contextBatch)):

            centerId = self.w2id[centerBatch[q]]
            h = self.W[centerId]

            contextId = self.w2id[contextBatch[q]]
            hc = self.Wp[contextId]

            score = self.clippedSigm(h @ hc.T)
            if self.monitor_results:
                loss -= np.log(score)
                loss_pos -= np.log(score)

            gradW[centerId] += (score - 1) * hc
            gradWp[contextId] += (score - 1) * h

            for k in range(negWinSize):
                n = self.w2id[
                    negativeBatch.pop()]  # Deletes the negative batch increasingly. A new negative batch is created each epoch
                hNeg = self.Wp[n]

                score = self.clippedSigm(hNeg @ h.T)
                if self.monitor_results:
                    _loss = np.log(self.clippedSigm(-hNeg @ h.T)) / negWinSize
                    loss -= _loss
                    loss_neg -= _loss

                gradW[centerId] += score * hNeg / negWinSize
                gradWp[n] += score * h / negWinSize

        if self.monitor_results:
            return loss, gradW, gradWp, loss_pos, loss_neg
        else:
            return 0, gradW, gradWp, 0, 0

    def backward(self, lr, gradW, gradsWp):

        self.W -= lr * gradW
        self.Wp -= lr * gradsWp

    def train(self, epochs, lr, batchSize, path):

        """

        :param trainset: list of all (center word, context word). One context word at a time. Can be shuffled
        :param batchSize:
        :param lr: learning rate
        :param path: path to save the model's weights after each epoch
        :return:
        """

        start_time = time.time()
        self.loss_history = []
        self.posloss_history = []
        self.negloss_history = []


        for epoch in range(epochs):

            batchIndices = list(range(0, len(self.trainset), batchSize))

            accloss = 0
            loss_pos = 0
            loss_neg = 0
            counter = 0

            for q, batchIndice in enumerate(batchIndices):

                centerBatch = self.trainset[batchIndice:batchIndice + batchSize, 0]  # List of center words [str]
                contextBatch = self.trainset[batchIndice:batchIndice + batchSize, 1]  # List of context words [str]
                negativeBatch = self.select_negative_sampling(self.neg_distrib,
                                                              batchSize * self.winNegSize)  # List of negative words [str]

                loss, gradW, gradsWp, lp, ln = self.forward(centerBatch, contextBatch, negativeBatch, self.winNegSize)

                if self.monitor_results:
                    accloss += loss
                    loss_pos += lp
                    loss_neg += ln
                    counter += batchSize

                self.backward(lr, gradW, gradsWp)

                if self.monitor_results and q % 100 == 0:
                    print("Epoch {} | Word Count :{} | Loss : {:.3f} | Pos/Neg : {:.3f}/{:.3f}".format(
                        epoch + 1, counter, accloss / counter, loss_pos / counter, loss_neg / counter))

                if self.monitor_results and q % 1000 == 0:
                    pass
                    #print("Similarity of Monday - Tuesday : {} / Monday - Financial : {}".format(self.similarity("monday", "tuesday"), self.similarity("monday", "financial")))

            if self.monitor_results:
                print("Epoch {} | Word Count :{} | Loss : {:.3f} | Neg/Pos : {:.3f}/{:.3f} | Eta : {:.3f}s".format(epoch+1, counter, accloss / counter, loss_pos / counter, loss_neg / counter, (epochs-(epoch+1))*((time.time()-start_time)/(epoch+1))))
                self.loss_history.append(accloss / counter)
                self.posloss_history.append(loss_pos / counter)
                self.negloss_history.append(loss_neg / counter)
                self.save(os.path.join(self.output_path, "model_epoch_{}".format(epoch)))
                np.save(os.path.join(self.output_path, "loss.npy"), np.array(self.loss_history))
                np.save(os.path.join(self.output_path, "posloss.npy"), np.array(self.posloss_history))
                np.save(os.path.join(self.output_path, "negloss.npy"), np.array(self.negloss_history))

            self.save(path)

    def save(self, path):
        """
        :param self:
        :param path: path to which the model is to be saved
        :return: None, saves the model's weighs to path
        """

        keys = np.array(list(self.w2id.keys()))
        ids = np.array(list(self.w2id.values()))
        model = np.vstack((keys, ids, self.W.T)).T

        with open(path, 'wb') as fp:
            pickle.dump(model, fp)

    @staticmethod
    def load(path):
        with open(path, 'rb') as file:
            model = pickle.load(file)
        return model
    """
    def load(self, path):
        

        :param self:
        :param path: path to the model
        :return: None, initiates the model with all the weighs saved previously
        

        
        with open(path, 'rb') as fp:
            model = pickle.load(fp)
            self.w2id = {model[0,k] : model[1,k] for k in range(model.shape[0])}
            self.W = model[:, 2:]
            
    """



if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--text', help='path containing training data', required=True)
    parser.add_argument('--model', help='path to store/read model (when training/testing)', required=True)
    parser.add_argument('--test', help='enters test mode', action='store_true')

    opts = parser.parse_args()

    if not opts.test:
        sentences = text2sentences(opts.text)
        n = len(sentences)

        if len(sentences) <= 1000:
            sg = SkipGram(sentences, nEmbed = 50, negativeRate = 0.01, winSize = 5, winNegSize = 5, minCount = 0, maxFreq = 1,
            monitor_results = False, experience_tag = "")
            sg.train(epochs = 100, batchSize = 16, lr = 0.05, path = opts.model)
        if 1000 < len(sentences) and len(sentences) <= 100000:
            sg = SkipGram(sentences, nEmbed = 150, negativeRate = 0.001, winSize = 5, winNegSize = 5, minCount = 5, maxFreq = 1,
            monitor_results = False, experience_tag = "")
            sg.train(epochs = 150, batchSize = 128, lr = 0.05, path = opts.model)
        if len(sentences) > 100000:
            sg = SkipGram(sentences, nEmbed = 300, negativeRate = 0.0001, winSize = 5, winNegSize = 2, minCount = 20, maxFreq = 1,
            monitor_results = False, experience_tag = "")
            sg.train(epochs = 200, batchSize = 256, lr = 0.01, path = opts.model)


    else:
        pairs = loadPairs(opts.text)

        model = SkipGram.load(opts.model)
        w2id = {model[k, 0]:model[k, 1] for k in range(model.shape[0])}
        W = model[:, 2:].astype(np.float)

        for a, b, _ in pairs:
            # make sure this does not raise any exception, even if a or b are not in sg.vocab
            print(similarity(a, b, w2id, W))
