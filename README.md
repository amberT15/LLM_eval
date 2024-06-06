# gLM evaluation analysis pipeline

This repository contains code to generate resutls for the study "Evaluating the representational power of pre-trained
DNA language models for regulatory genomics". ([Pre-print link](https://www.biorxiv.org/content/10.1101/2024.02.29.582810v1))



The `data_generation` folder contains script for the pre-processing of datsets, and notebooks of using each gLM to exract layer embeddings. `figure` contains code and generated figure for the paper.

The rest of the code is orgnized by task and analysis:
* `lentiMPRA` for Task 1
* `chip-clip-seq` for Task 2 and 6
* `CAGI` for Task 3
* `alternative-splicing` for Task 4
* `RNAenlong` for Task 5
* `motif_id` for all saliency analysis

Within each repository are orgnized based on the input. Most folders contain scripts for gLM representation (except NT), NT, and one-hot based model trainings. 

Since not all gLMs can be installed in the same environment, three different environments were used during this study, `tf_requirments.yml`, `torch_requirments.yml` and `gpn_requirements.yml`. 
* `tf_requirments` will be most frequently used. This environment should be used for all scripst based on Nucleotide Transformer (`NT_*`), and also the onehot and representation based model trainings(`lentiMPRA/representation_perf.ipynb`,`lentiMPRA/onehot_models.py` etc.).
* `torch_requirments` are used for HyenaDNA based scripts. Mostly in `data_generation/embedding_generation/Heyna_embed.ipynb` and `CAGI/cagi_NT.ipynb`
* and `gpn_requirments` are for all GPN related inference, such as `data_generation/embedding_generation/GPN_embed.ipynb` and `CAGI/cagi_gpn.ipynb`


Original dataset and models trained for this study can be accessed from [zenodo](https://doi.org/10.5281/zenodo.8279716), they should be decompressed into the base folder for this repo. No installation is required to run analysis in this repository

