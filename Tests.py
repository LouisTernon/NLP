from skipGram import *
import pickle
import os




path = 'sentences_100.pickle'

if os.path.isfile(path):
    sentences = pickle.load((open(path, 'rb')))
else:
    sentences = text2sentences(
        "news.en-00023-of-00100",
        100)
    with open(path, 'wb') as f:
        pickle.dump(sentences, f)


sg = SkipGram(sentences, minFreq=1e-4, maxFreq=1e-2, negativeRate=1e-2, nEmbed=50, winSize=1, winNegSize=1, monitor_results=True, experience_tag="100sentences50emb1winSize1winNeg")

sg.train(epochs=1000, lr=0.01, batchSize=8)