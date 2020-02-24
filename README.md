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
We identified various combinations of parameters based on the size of the corpus.

The default parameters vary depending on the length of the dataset : 
- For 1000 lines or less:
    - Embedding dimension = 50
    - Negative rate = 1e-2  // maximal frequency for a word to qualify to be negatively sampled
    - Context Size = 5
    - Minimum Count = 0
    - Negative Size = 5 // number of negatively sampled words per context word
    - maxFreq = 1
    - Epochs = 100
    - Batch Size = 16
    - Learning Rate = 0.05

- For 1001 to 100,000 lines: 
    - Embedding dimension = 150
    - Negative rate = 1e-3  // maximal frequency for a word to qualify to be negatively sampled
    - Context Size = 5
    - Minimum Count = 5
    - Negative Size = 5 // number of negatively sampled words per context word
    - maxFreq = 1
    - Epochs = 150
    - Batch Size = 128
    - Learning Rate = 0.05

- For more than 100,000 lines: 
    - Embedding dimension = 300
    - Negative rate = 1e-4  // maximal frequency for a word to qualify to be negatively sampled
    - Context Size = 5
    - Minimum Count = 5
    - Negative Size = 2 // number of negatively sampled words per context word
    - maxFreq = 1
    - Epochs = 200
    - Batch Size = 256
    - Learning Rate = 0.025


## Though process

Our though process can be found in the report attached, especially in the fifth section, challenges. In summary, we first implemented a vanilla skip-gram, taking inspiration from [X. Rong](https://arxiv.org/abs/1411.2738, 201) and [Y. Goldberg](https://arxiv.org/abs/1402.3722), have further information about the optimization process, how to perform the updates of the embeddings, and gain a finer understanding of negative sampling particularities. Our first models gave poor performances, and especially a recurrent behavior was the very rapid convergence of all embedding to a unique value. At the time, we were running experiments on very small corpuses, to get faster insight. It was difficult to undestands what caused the model to behave that way. Going deeper in the litterature, we mostly identified two potential issues: the first being the quality of the data (since we used small corpuses, the information is very noizy, negative sampling distribution not meaningfull, ..), so we tried multiple strategies to enhance the quality of the data (Lossy counting, subsampling high frequency, ..), and used bigger corpuses. The second source was the model itself, since we believed skipgram negative sampling wasn't powerfull enough, leading to all embedding being similar. The rate of experiments was very slow due to the important training time of the models, and it did not solve the issue. We eventually realized that, using our settings, negative sampling was actually too powerful, causing the positive loss to increase and converging to this wrong solution. We managed to fix this issue by averaging the loss of negative sampling for each context words, allowing us to have models converging to actual optimal solutions. However, the embeddings are not yet significant, since we were not able to train the model on important corpuses (>10k sentences). 