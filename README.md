## FERMAT: An Alternative to Accuracy for Numerical Reasoning

This is the repository for <a href="https://aclanthology.org/2023.acl-long.838/">FERMAT: An Alternative to Accuracy for Numerical Reasoning</a> by <a href="https://jasivan.github.io/">Jasivan Alex Sivakumar</a> and <a href="https://ns-moosavi.github.io/">Nafise Sadat Moosavi</a> accepted at ACL 2023.

You can find the codes, templates and test sets in their respective folders.

## Usage
<ul>
  <li>Codes contains the codes for training and evaluating the model.</li>
  <li>Templates contains the templates written by expert maths teachers, from GSM8K and AQUA too.</li>
  <li>Test_sets contains the Original test set (based on [Illinois](https://aclanthology.org/N16-3011.pdf) and [CommonCore](https://aclanthology.org/D15-1202.pdf)) and the 18 variation test sets used in FERMAT.</li>
</ul>

### Requirements
Install the packages from the requirements.txt file (transformers, torch, datasets, accelerate, SentencePiece, json, pandas, numpy).

## Reference
If you use this code in your work, please cite the paper:
```
@inproceedings{sivakumar-moosavi-2023-fermat,
    title = "{FERMAT}: An Alternative to Accuracy for Numerical Reasoning",
    author = "Sivakumar, Jasivan  and
      Moosavi, Nafise Sadat",
    editor = "Rogers, Anna  and
      Boyd-Graber, Jordan  and
      Okazaki, Naoaki",
    booktitle = "Proceedings of the 61st Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers)",
    month = jul,
    year = "2023",
    address = "Toronto, Canada",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2023.acl-long.838/",
    doi = "10.18653/v1/2023.acl-long.838",
    pages = "15026--15043"
```

## Authors
This code was written by [@jasivan](https://github.com/jasivan/).

If you have any queries or comments, please email <a href="mailto: jasivakumar1@sheffield.ac.uk"> jasivakumar1@sheffield.ac.uk </a>.
