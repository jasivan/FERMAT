The codes for <a href="https://arxiv.org/abs/2305.17491">FERMAT: An Alternative to Accuracy for Numerical Reasoning</a> are being updated. Please come back for more updated codes.

## Usage
Basic usage with CoNLL files:

	$ python finetune.py --model_checkpoint --route --output

`model_checkpoint` is where the model's checkpoint is located or the huggingface model name
`route` is the path to the train/dev/test sets
`output` is the path to the save the output files