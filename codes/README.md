The codes for <a href="https://arxiv.org/abs/2305.17491">FERMAT: An Alternative to Accuracy for Numerical Reasoning</a> are being updated. Please come back for more updated codes.

## Usage
Basic usage with CoNLL files:

	$ python finetune.py --model_checkpoint --route --output

`model_checkpoint` is where the model's checkpoint is located or the huggingface model name
`route` is the path to the train/dev/test sets
`output` is the path to the save the output files

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