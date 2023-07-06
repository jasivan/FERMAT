The codes for <a href="https://arxiv.org/abs/2305.17491">FERMAT: An Alternative to Accuracy for Numerical Reasoning</a> are being updated. Please come back for more updated codes.

## Usage
Basic usage to finetune the model on a given training and development set:

	$ python finetune.py --model_checkpoint --route --output

`model_checkpoint` is where the model's checkpoint is located or the huggingface model name <br>
`route` is the path to a folder with train/dev/test sets <br>
`output` is the path to the save the output files

Basic usage to generate evaluate from saved predictions:

	$ python evaluate_from_ckpt.py --model_checkpoint --evaluation_set --output --prompt --digit

`model_checkpoint` is where the model's checkpoint is located or the huggingface model name <br>
`evaluation_set` is the path to evaluation set as json <br>
`output` is the path to the save the results <br>
`prompt` is the prompt used for the input <br>
`digit` is whether you want digit tokenisation or not