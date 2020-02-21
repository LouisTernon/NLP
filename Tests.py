from skipGram import *


sentences = text2sentences("1-billion-word-language-modeling-benchmark-r13output/training-monolingual.tokenized.shuffled/news.en-00023-of-00100")

sg = SkipGram(sentences[:10000], minCount=10, negativeRate=0.005, nEmbed=50)



# Le gradient exploseeee
sg.train(epochs=10, lr=0.01)

sg.save("")


sg.load("")

print(sg.similarity("king", "queen"))

print(sg.W.shape)