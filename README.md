This is a repository for reproducing the ACL 2023 paper: Verify-and-Edit: A Knowledge-Enhanced Chain-of-Thought Framework.

Each folder contains the scripts and data for reproducing the experiments. Running commands are in written in commands.md.


# Citation

```
@inproceedings{zhao-etal-2023-verify,
    title = "Verify-and-Edit: A Knowledge-Enhanced Chain-of-Thought Framework",
    author = "Zhao, Ruochen  and
      Li, Xingxuan  and
      Joty, Shafiq  and
      Qin, Chengwei  and
      Bing, Lidong",
    editor = "Rogers, Anna  and
      Boyd-Graber, Jordan  and
      Okazaki, Naoaki",
    booktitle = "Proceedings of the 61st Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers)",
    month = jul,
    year = "2023",
    address = "Toronto, Canada",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2023.acl-long.320",
    doi = "10.18653/v1/2023.acl-long.320",
    pages = "5823--5840",
    abstract = "As large language models (LLMs) have become the norm in NLP, demonstrating good performance in generation and reasoning tasks, one of its most fatal disadvantages is the lack of factual correctness. Generating unfactual texts not only leads to lower performances but also degrades the trust and validity of their applications. Chain-of-Thought (CoT) prompting improves trust and model performance on complex reasoning tasks by generating interpretable reasoning chains, but still suffers from factuality concerns in knowledge-intensive tasks. In this paper, we propose the Verify-and-Edit framework for CoT prompting, which seeks to increase prediction factuality by post-editing reasoning chains according to external knowledge. Building on top of GPT-3, our framework lead to accuracy improvements in multiple open-domain question-answering tasks.",
}
```


