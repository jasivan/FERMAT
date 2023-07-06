# FERMAT: An Alternative to Accuracy for Numerical Reasoning

This is the repository for <a href="https://arxiv.org/abs/2305.17491">FERMAT: An Alternative to Accuracy for Numerical Reasoning</a> by <a href="https://jasivan.github.io/">Jasivan Alex Sivakumar</a> and <a href="https://ns-moosavi.github.io/">Nafise Sadat Moosavi</a> accepted at ACL 2023.

You can find the codes and test sets in their respective folders. The codes folder is still being updated


### Requirements
Install the packages from the requireents.txt file (transformers, torch, datasets, accelerate, SentencePiece).

## Usage
Basic usage with CoNLL files:

	$ python scorer.py key system

`key` and `system` are the files with gold coreference and system output, respectively.

For more details, refer to
[ARRAU README](https://github.com/ns-moosavi/coval/blob/master/arrau/README.md)
for evaluations of the ARRAU files and
[CoNLL README](https://github.com/ns-moosavi/coval/blob/master/conll/README.md)
for CoNLL evaluations.

Run tests with `python3 -m pytest unittests.py`

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