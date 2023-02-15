#!/bin/bash
#SBATCH --mem=60G
#SBATCH --time=08:00:00
#SBATCH --gpus=1
#SBATCH --mail-user=jasivakumar1@sheffield.ac.uk
#SBATCH --partition=dcs-gpu
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --account=dcs-res

module load Anaconda3/5.3.0
source activate JAS
python T5_code/summarizer_token.py \
--model_checkpoint='facebook/bart-large' \
--route=data/AQUA/json \
--epochs=100 \
--output=../../../fastdata/acp21jas/results/tokenised/AQUA/bart-large-q

module load Anaconda3/5.3.0
source activate JAS
python T5_code/summarizer_token2.py \
--model_checkpoint='facebook/bart-large' \
--route=data/AQUA/json \
--epochs=100 \
--output=../../../fastdata/acp21jas/results/tokenised/AQUA/bart-large-q-a

module load Anaconda3/5.3.0
source activate JAS
python T5_code/summarizer.py \
--model_checkpoint='facebook/bart-large' \
--route=data/AQUA/json \
--epochs=100 \
--output=../../../fastdata/acp21jas/results/tokenised/AQUA/bart-large-untokenised

module load Anaconda3/5.3.0
source activate JAS
python T5_code/summarizer_token.py \
--model_checkpoint='t5-large' \
--route=data/AQUA/json \
--epochs=100 \
--output=../../../fastdata/acp21jas/results/tokenised/AQUA/t5-large-q

module load Anaconda3/5.3.0
source activate JAS
python T5_code/summarizer_token2.py \
--model_checkpoint='t5-large' \
--route=data/AQUA/json \
--epochs=100 \
--output=../../../fastdata/acp21jas/results/tokenised/AQUA/t5-large-q-a

module load Anaconda3/5.3.0
source activate JAS
python T5_code/summarizer.py \
--model_checkpoint='t5-large' \
--route=data/AQUA/json \
--epochs=100 \
--output=../../../fastdata/acp21jas/results/tokenised/AQUA/t5-large-untokenised