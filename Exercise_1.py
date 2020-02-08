
from __future__ import division
import argparse
import pandas as pd

# useful stuff
import numpy as np
from scipy.special import expit
from sklearn.preprocessing import normalize


__authors__ = ['author1','author2','author3']
__emails__  = ['fatherchristmoas@northpole.dk','toothfairy@blackforest.no','easterbunny@greenfield.de']

def text2sentences(path):
	# feel free to make a better tokenization/pre-processing

	"""
	:param path: path to the text files
	:return: list of all sentences
	"""


	sentences = []
	with open(path) as f:
		for l in f:
			sentences.append( l.lower().split() )
	return sentences

def loadPairs(path):

	"""

	:param path: path to the csv of word pairs
	:return: zip object of word pairs
	"""


	data = pd.read_csv(path, delimiter='\t')
	pairs = zip(data['word1'],data['word2'],data['similarity'])
	return pairs


class SkipGram():
	def __init__(self, sentences, nEmbed=100, negativeRate=0.2, winSize = 5, winNegSize = 5, minCount = 5):
		self.valid_vocab, self.vocab_occ, self.eliminated_words = self.initialize_vocab(sentences, minCount)
		# set of valid words  over the whole dataset, dict of their occurences, and set of all eliminated words
		self.w2id, self.id2w = self.word_id() # word to ID mapping
		self.trainset = self.filter_train_set(self.eliminated_words, sentences) # set of sentences
		self.neg_vocab_set, self.n_neg = self.build_neg_vocab(self.valid_vocab, self.vocab_occ, negativeRate)
		self.neg_distrib = self.build_neg_distrib(self.neg_vocab_set)
		self.nEmbed = nEmbed
		self.n_words = len(self.valid_vocab)# number of words that appear at list five times

		self.winSize = winSize # Window size
		self.winNegSize = winNegSize # Number of negative words per target vector

		#Weighs of the model
		#self.W = np.random.random((self.n_words, nEmbed))
		#self.Wp = np.random.random((nEmbed, self.n_words))


	def initialize_vocab(self, sentences, minCount):
		all_words  = [word for sentence in sentences for word in sentence] # list of all the words of all sentences
		all_words_unique = set(all_words)
		vocab_occ = {word:0 for word in all_words_unique}
		for word in all_words:
			vocab_occ[word] += 1
		valid_words_set = set(filter(lambda x: vocab_occ[x] > minCount, all_words_unique))
		invalid_words_set = all_words_unique - valid_words_set
		return valid_words_set, vocab_occ, invalid_words_set

	def word_id(self):
		"""
		:param setntences: set of all words in the vocab, occ > minCount
		:return: dictionaries {word:id} and {id:word}
		"""
		return {word:q for q, word in enumerate(self.valid_vocab)}, \
			   {q:word for q, word in enumerate(self.valid_vocab)}

	def filter_train_set(self, invalid_vocab, sentences):
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
			for p in to_del[::-1]: # reverse index to be able to use del
				del sentences[q][p]

		return sentences

	def build_neg_vocab(self, vocab, vocab_occ, neg_treshold):
		"""

		:param vocab: all the word that will be in the train set and occur at least minCount times
		:param vocab_occ: dict where {valid word  : number of occurences}
		:param neg_treshold: maximum time a word can occur to be qualified for the negative words set
		:return:
		"""

		n = np.sum([vocab_occ[word]  for word in vocab])
		vocab_freq = {word:vocab_occ[word]/n for word in vocab}
		neg_vocab = list(filter(lambda x:vocab_freq[x] < neg_treshold, vocab))
		neg_vocab_set = set(neg_vocab)
		n_neg = len(neg_vocab)

		return neg_vocab_set, n_neg

	def build_neg_distrib(self, neg_vocab):

		"""
		TODO : We might need to add a transformation to the distribution of the negative words
		see http://mccormickml.com/2017/01/11/word2vec-tutorial-part-2-negative-sampling/ > SELECTING NEGATIVE SAMPLE


		:param neg_vocab: list of all the words that qualify for the negative vocabulary
		:return: a dictionary with the frequency of each word of the negative vocabulary
		"""
		n = np.sum([self.vocab_occ[word]  for word in neg_vocab])
		distr = {neg_w : self.vocab_occ[neg_w]/n for neg_w in neg_vocab}

		return distr


	def transform_34(self, freqw, n):
		return (freqw/n) **0.75


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
			context_before = sentence[k-K:k]
			context_after = sentence[k+1:k+1+K]
			contexts.append(context_before + context_after)
			center_words.append(word)
		return center_words, contexts


	def select_negative_sampling(self, neg_distribution, K):
		"""

		:param neg_vocab: set of the negative vocab
		:param neg_distribution: {neg_word : sample rate}
		:return: returns K negative words
		"""

		return [np.random.choice(list(neg_distribution.keys()), p = list(neg_distribution.values())) for x in range(K)]

	def oneHot(self, dim, id):
		oh = np.zeros(dim, int)
		oh[id] = 1
		return oh


	def forward(self, center_word, target_context_word, neg_words):

		"""

		:param self:
		:param center_word: word of the center context to evaluate
		:param target_context_word: one word of the context
		:param neg_words: list of negative words
		:return: loss function and intermediate gradients
		"""
		h1 = self.W[self.w2id[center_word], :]  	# output of the first layer for vcenter_word

		h2 = h1.T @ self.Wp

		h2_target = (self.W[self.w2id[target_context_word], :]).T @ self.Wp

		h2_neg = []

		for neg_word in neg_words:
			h2_neg.append((self.W[self.w2id[neg_word], :]).T @ self.Wp)

		loss = -np.log(expit(h2.T @ h2_target)) - np.sum([np.log(expit(-h2_n.T @ h2_target)) for h2_n in h2_neg])

		gradPred = (expit(h2.T @ h2_target) - 1) * h2 - np.sum([expit(-h2_n.T @ h2_target)*h2_n for h2_n in h2_neg])
		#grad_Wp


		indice_center_grad = self.w2id[center_word]
		indices_for_grad = [self.w2id[word] for word in neg_words]

		grad = np.zeros(self.Wp.shape)

		for n, id in enumerate(indices_for_grad):
			gradNeg = - (expit(-h2_neg[n].T @ h2_target)-1)*h2_target
			grad[id] = gradNeg

		grad[indice_center_grad] = (expit(h2.T @ h2_target)-1)*h2_target

		#grad_W1

		return loss, gradPred, grad


	def backward(self, grad_pred, grad, lr):

		self.W -= lr* grad
		self.W2 -= lr*grad



	def train(self, epochs):

		loss = 0 # Initialise loss at 0

		self.W = np.random.normal(size = self.n_words * self.nEmbed).reshape(self.n_words, self.nEmbed)/np.sqrt(self.n_words)
		self.Wp = np.random.normal(size = self.nEmbed * self.n_words).reshape(self.nEmbed, self.n_words)/np.sqrt(self.n_words)
		self.accloss = 0
		self.counter = 0

		for epoch in range(epochs):

			for counter, sentence in enumerate(self.trainset):

				# Might use tqdm to show progress bar

				# sentence = filter(lambda word: word in self.vocab, sentence)
				# Already implemented during the initialisation of the class

				sentence_words, sentence_contexts = self.parse_sentence_for_context(sentence, self.winSize)

				for q, contexts in enumerate(sentence_contexts):

					for context_word in contexts:
						word = sentence_words[q] # Center word of the q-th sentence
						neg_words = self.select_negative_sampling(self.neg_distrib, self.winNegSize)
						if context_word == word: continue

						loss, grad_pred, grad = self.forward(word, context_word, neg_words)

						self.accLoss += loss

						self.backward(grad_pred, grad)

	def save(self,path):
		"""

		:param self:
		:param path: path to which the model is to be saved
		:return: None, saves the model's weighs to path
		"""
		raise NotImplementedError('implement it!')

	def load(self,path):
		"""

		:param self:
		:param path: path to the model
		:return: None, initiates the model with all the weighs saved previously
		"""
		raise NotImplementedError('implement it!')




	def similarity(self,word1,word2):
		"""
			computes similiarity between the two words. unknown words are mapped to one common vector
		:param word1:
		:param word2:
		:return: a float \in [0,1] indicating the similarity (the higher the more similar)
		"""
		a, b = self.w2id(word1), self.w2id(word2)
		e1, e2 = self.W[a,:], self.W[b,:]

		return e1.T @ e2 / (np.linalg.norm(e1) * np.linalg.norm(e2))

	@staticmethod
	def load(path):
		raise NotImplementedError('implement it!')



"""

if __name__ == '__main__':

	parser = argparse.ArgumentParser()
	parser.add_argument('--text', help='path containing training data', required=True)
	parser.add_argument('--model', help='path to store/read model (when training/testing)', required=True)
	parser.add_argument('--test', help='enters test mode', action='store_true')

	opts = parser.parse_args()

	if not opts.test:
		sentences = text2sentences(opts.text)
		sg = SkipGram(sentences)
		sg.train(...)
		sg.save(opts.model)

	else:
		pairs = loadPairs(opts.text)

		sg = SkipGram.load(opts.model)
		for a,b,_ in pairs:
			# make sure this does not raise any exception, even if a or b are not in sg.vocab
			print(sg.similarity(a,b))

"""