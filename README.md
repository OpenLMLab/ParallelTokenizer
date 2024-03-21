# ParallelTokenizer

### Latest News 🔥

- 2024/03/21: Added compatibility with ChatGLM3


## Introduction

ParallelTokenizer is an open-sourced efficient tokenization framework which aims to support the efficient tokenization of a context with thousands of or millions of tokens. The working principle of parallel tokenizer can be summarized as follows. 

1. **Split**. ParallelTokenizer first splits the input text based on a pre-defined chunk_size. Different from regular chunking, in order to avoid the token at the cutting point being destroyed, a piece of overlap needs to be reserved for the two slices before and after. The size of this overlap should be no less than the longest token in the vocabulary measured in char. By default, ```chunk_size=40960``` and ```overlap=512```.

2. **Tokenize**. After the input text being split, ParallelTokenizer conducts tokenization of different segment in parallel and arquires the token lists of different segments.

3. **Merge**. ParallelTokenizer finally merge the token lists of different segments with the overlap being evicted to obtain the token list of the entire input. Importantly, the overlap part can be located with LCS algorithm. ParallelTokenizer concatenate the token list before the overlap part in the previous chunk, the token in the overlap, and the token list after the overlap part in the next chunk, thus obtaining the result of tokenizing the entire input content.

<img src="figures/parallel_tokenizer.gif" width="63%" height="63%">

Use ParallelTokenizer can help you achieve superior acceleration, compared with the common tokenization when you processing the tokenization of extremely long context.

<img src="figures/tokenizer_speed.png" width="63%" height="63%">
