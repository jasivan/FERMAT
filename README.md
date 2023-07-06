# FERMAT: An Alternative to Accuracy for Numerical Reasoning

This is the repository for <a href="https://arxiv.org/abs/2305.17491">FERMAT: An Alternative to Accuracy for Numerical Reasoning</a> by <a href="https://jasivan.github.io/">Jasivan Alex Sivakumar</a> and <a href="https://ns-moosavi.github.io/">Nafise Sadat Moosavi</a> accepted at ACL 2023.

You can find the codes, templates and test sets in their respective folders. Note that the codes folder is still being updated.

## Usage
Codes contains the codes for training and evaluating the model.<br>
Templates contains the templates written by expert maths teachers, from GSM8K and AQUA too. <br>
Test_sets contains the Original test set (based on [Illinois](https://aclanthology.org/N16-3011.pdf) and [CommonCore](https://aclanthology.org/D15-1202.pdf)) and the 18 variation test sets used in FERMAT.

### Requirements
Install the packages from the requirements.txt file (transformers, torch, datasets, accelerate, SentencePiece, json, pandas, numpy).

## Reference
If you use this code in your work, please cite the paper:
```
@InProceedings{sivakumar-moosavi_2023_FERMAT,
    author = { Jasivan Alex Sivakumar, Nafise Sadat Moosavi},
    title = {FERMAT: An Alternative to Accuracy for Numerical Reasoning},
    year = {2023},
    booktitle = {Proceedings of the 61st Annual Meeting of
		the Association for Computational Linguistics (Volume 1: Long Papers)},
    publisher = {Association for Computational Linguistics},
    address = {Toronto, Canada},
}
```

## Authors
This code was written by [@jasivan](https://github.com/jasivan/).

If you have any queries or comments, please email <a href="mailto: jasivakumar1@sheffield.ac.uk"> jasivakumar1@sheffield.ac.uk </a>.