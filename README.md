# CopperMT - Cognate Prediction per MT
This repository contains the code for ACL 2021 Findings paper: Can Cognate Prediction Be Modelled as a Low-Resource Machine Translation Task?

## Overview
We provide a pipeline, based on fairseq, to train bilingual or multilingual NMT models, with either pretraining or backtranslation. They can also be compared to SMT models (using MOSES). The scripts can be used as such to reproduce our results, or modified to fit your analyses.

Our results on cognate prediction for some Romance languages, when comparing multilingual transformer and RNN:
![](https://clefourrier.github.io/img/papers/ACL2021.png)

## Repository organisation
- inputs
   - raw_data and split_data (contains bilingual aligned datasets or monolingual source2source or target2target datasets)
   - parameters (parameter files for the models)
- pipeline
   - data (extractor for EtymDB, dataraw on raw aligned daa, data on split aligned data)
   - neural_translation (scripts to use our Multi Encoder multi Decoder Architecture, MEDeA) 
   - statistical translation (scripts to use and finetune MOSES)
   - utils (for bleu use)
   - various mains and parameters.cfg

## How to use
Remark: This code has been tested on Unix-like systems (MacOS, Ubuntu, Manjaro).

### 1) Install the requirements
#### requirements.txt
You can create a virtualenv
```bash 
virtualenv -p python3 pyenv
source pyenv/bin/activate
pip install -r requirements.txt
```
If you intend to extract and phonetize data, pleae install espeak manually

#### git submodules (optional)
If you want to do SMT (and use MOSES and mgiza) or extract cognate data yourself (using EtymDB), you will need the submodules. You can skip MOSES install if you already installed it somewhere else on your machine.

1) Install boost (>1.64), and follow the documentation of [MOSES](https://www.statmt.org/moses/?n=Development.GetStarted) regarding extra packages you might need depending on your distribution
2) Then, initialize the submodules using
```bash
git submodule init
git submodule update
```
3) Finish the install of mgiza: `cd submodules/mgiza/mgizapp; cmake .; make; make install`. The MGIZA binary and the script merge_alignment.py need to be copied in your binary directory that Moses will look up for word alignment tools (in our case, submodule/mgiza/mgizapp/bin) `cp scripts/merge_alignment.py bin/`
4) Finish the install of moses: `cd ../../mosesdecoder; bjam -j4 -q -d2` (if on mac you might need to checkout the branch clang-error and correct the errors during the bjam build)

### 2) Edit the parameter files
Edit your parameter files, change MEDeA_DIR to the path of your installation.

### 3) Reproduce the paper results: launch the scripts
```bash
cd pipeline
bash main_<your script of choice>.sh parameters.cfg
```

## Licence
All code here is mine (clefourrier) except for the spm_train.py script
(in pipeline/neural_translation/) which comes from fairseq (under
MIT licence) and has been added here for convenience. My code is under GNU GPL 3.

## Attribution
If you use the code, models or algorithms, please cite:
```
@inproceedings{fourrier-etal-2021-cognate,
    title = "Can Cognate Prediction Be Modelled as a Low-Resource Machine Translation Task?",
    author = "Fourrier, Cl{\'e}mentine  and
      Bawden, Rachel  and
      Sagot, Beno{\^\i}t",
    booktitle = "Findings of the Association for Computational Linguistics: ACL-IJCNLP 2021",
    month = "aug",
    year = "2021",
    address = "Online",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2021.findings-acl.75",
    doi = "10.18653/v1/2021.findings-acl.75",
    pages = "847--861",
}

```
