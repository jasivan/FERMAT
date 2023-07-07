The test sets are split into questions with a .q extension and answers with .a extension. They are indexed so that they can be paired easily. This is because we aim to reduce the amount of possible leakage when large language models are training on public github repositories.

## Usage
Basic usage to generate the .test files that match the questions and answers into a single json file:

	$ python match_q_a.py