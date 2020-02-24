# First Exercise – Skip-gram with negative-sampling from scratch

## Run
To train:
```bash
python skipGram.py --text <path_to_trainset> --model <model.model>
```
To test:
```bash
python skipGram.py --text <relative_path_to_EN-SIMLEX-999.txt> --model <model.model> --test
```

## Default parameters
The default parameters vary depending on the length of the dataset : 
For 1000 lines or less:
- Embedding dimension = 50
- Negative rate = 1e-2  // maximal frequency for a word to qualify to be negatively sampled
- Context Size = 5
- Minimum Count = 0
- Negative Size = 5 // number of negatively sampled words per context word
- maxFreq = 1
- Epochs = 100
- Batch Size = 16
- Learning Rate = 0.05

For 1001 to 100,000 lines: 
- Embedding dimension = 150
- Negative rate = 1e-3  // maximal frequency for a word to qualify to be negatively sampled
- Context Size = 5
- Minimum Count = 5
- Negative Size = 5 // number of negatively sampled words per context word
- maxFreq = 1
- Epochs = 150
- Batch Size = 128
- Learning Rate = 0.05

For more than 100,000 lines: 
- Embedding dimension = 300
- Negative rate = 1e-4  // maximal frequency for a word to qualify to be negatively sampled
- Context Size = 5
- Minimum Count = 5
- Negative Size = 2 // number of negatively sampled words per context word
- maxFreq = 1
- Epochs = 200
- Batch Size = 256
- Learning Rate = 0.025


## Model Architecture 
The word embedings derive from the skip-gram model whose objective is to predict a word's context. To train the model, we must predict the context of each word of the training set. It is like a sliding window on every word of the sentences. 
The back propagation of such model is very costly and requires to make computations for each possible word. The loss function needs to be penalised with all the words that are not in the context. Hence the negative sampling, a method that enables penalisation of the skip gram with random words. We do the hypothesis that random word are related nor to the center word, nor to the context word. This allow to only compute a few words (~10), compared with (~1e3 - 1e6) with the basic skip gram. 

## Modifications and choices 
We had a hard time getting the model to converge. At the beggining, we only had one epoch (like in Mikolov's code in C). This led to having all the word embeddings colinear (similarity returned .99). However, by doing several epochs, we were able to lower the loss. We noted that the loss decreased after less epochs for bigger models. Mikolov used a much bigger corpus than us. It might be the reason why he only did one epoch. 
In parralel, we introduced batch sizes and shuffled the train set to have a more stable convergence. We don't think we need to explain the batch size. Regarding the shuffling however, here is how we did it : We generated pairs of (center word, context word). One context word at a time for each center word. The pairs were then shuffled. The negatively word were all sampled at the same time to improve computation time. If the context word or the center word were the same as a negative sample, they were not calculated. Last, in order to stablise the convergence, we clipped the gradients to avoid overflow. 
In order to improve computation time, we gave the model's W and W' forward matrices the same dimension (W.T should have the same shape as W'). So, we only needed to access the rows of W and W', instead of their columns. 
