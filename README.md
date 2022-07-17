# PIECER

## Introduction

This project contains the main code to train and evaluate four base MRC models (QANet, BERT-base, BERT-large, RoBERTa-base) + PIECER. 

## Usage

For QANet + PIECER, run `bash run_qanet.sh`

For BERT-base + PIECER, run `bash run_bert.sh`

For BERT-large + PIECER, run `bash run_bertl.sh`

For RoBERTa-base + PIECER, run `bash run_roberta.sh`

Hyper-parameters can be manually set in each shell script. 

## Data

ReCoRD can be downloaded from its [homepage](https://sheng-z.github.io/ReCoRD-explorer/) or the [SuperGLUE download link](https://super.gluebenchmark.com/tasks). 

ConceptNet can be downloaded from its [download link](https://github.com/commonsense/conceptnet5/wiki/Downloads). 

Our used data are totally oriented from these two sources. We will also release the preprocessing scripts and the preprocessed data later. 

## Citation

If you use this code for your research, please kindly cite our NLPCC-2022 paper:
```
@inproceedings{dai2022piecer,
  author    = {Damai Dai and
               Hua Zheng and
               Zhifang Sui and
               Baobao Chang},
  title     = {Plug-and-Play Module for Commonsense Reasoning in Machine Reading Comprehension},
  booktitle = {Proceedings of the 11th CCF International Conference on Natural Language Processing and Chinese Computing, {NLPCC} 2022, Guilin, China, September 22-25, 2022},
  year      = {2022},
}
```

## Contact

Damai Dai: daidamai@pku.edu.cn
