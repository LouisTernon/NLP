from skipGram import *
import pickle
import os




path = 'sentences.pickle'

if os.path.isfile(path):
    sentences = pickle.load((open('sentences.pickle', 'rb')))
else:
    sentences = text2sentences(
        "1-billion-word-language-modeling-benchmark-r13output/training-monolingual.tokenized.shuffled/news.en-00023-of-00100",
        10000)
    with open('sentences.pickle', 'wb') as f:
        pickle.dump(sentences, f)
    


sg = SkipGram(sentences, minCount=10, negativeRate=1e-2, nEmbed=50, winSize=5, winNegSize=5)



sg.train(epochs=10, lr=0.05, batchSize=256)