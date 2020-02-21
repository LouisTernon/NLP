from skipGram import *


sentences = text2sentences("1-billion-word-language-modeling-benchmark-r13output/training-monolingual.tokenized.shuffled/news.en-00023-of-00100")

sg = SkipGram(sentences[:10000], minCount=10, negativeRate=1, nEmbed=100, winSize=5, winNegSize=10)



# Le gradient exploseeee
sg.train(epochs=10, lr=0.002)

sg.save("")


sg.load("")

print(sg.similarity("king", "queen"))

print(sg.W.shape)