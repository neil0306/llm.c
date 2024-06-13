# GPT训练常见数据集

GPT-2 和 GPT-3 文章中, 里面有个`Common Crawl` 数据集, 这个数据集包含很多 random 数据, noise 特别多, 感觉不是很有用.

比较优质的数据集:
- Red Pajama 数据集
  - 子集 Slim Pajama: https://www.cerebras.net/blog/slimpajama-a-627b-token-cleaned-and-deduplicated-version-of-redpajama
    - 包含 627B tokens
    - 经过了数据清洗, 去重  
- FineWeb 数据集
  - HuggingFace 链接: https://huggingface.co/datasets/HuggingFaceFW/fineweb
    - more than 15T tokens (36.7TB) of cleaned and deduplicated English web data from Common Crawl
  - FineWeb-Edu
    - FineWeb的子集, 有 1.5T token 和 5.4T token 版本
    - https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu

