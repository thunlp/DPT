# DPT

Source code and dataset for ACL 2022 Findings paper [Prompt Tuning for Discriminative Pre-trained Language Models](https://aclanthology.org/2022.findings-acl.273.pdf)

Recent works have shown promising results of prompt tuning in stimulating pre-trained language models (PLMs) for natural language processing (NLP) tasks. In this work, we present DPT, the first prompt tuning framework for discriminative PLMs, which reformulates NLP tasks into a discriminative language modeling problem. Comprehensive experiments on text classification and question answering show that, compared with vanilla fine-tuning, DPT achieves significantly higher performance, and also prevents the unstable problem in tuning large PLMs in both full-set and low-resource settings.

### Computing infrastructure

##### OS:

Distributor ID: Ubuntu
Description:    Ubuntu 16.04.1 LTS
Release:    16.04
Codename:   xenial

##### GPU:

GeForce RTX 2080Ti

##### Language:

Python 3.7.5

##### Required packages:

For Text Classification, you can install with:

```
cd src/TextClassification/ 
pip install -r requirements.txt
```

For Question Answering, you can build by docker with:

```
cd src/QuestionAnswering/
docker build -t dpt:v0 .
docker run --gpus '"device=0,1,2,3,4,5"' -it -v [your path]/src/QuestionAnswering:/QuestionAnswering  --name=DPT_QA dpt:v0
```


### Run

##### Text Classification

```
cd src/TextClassification/
bash run_textclassification.sh
```

##### Question Answering

```
docker attach DPT_QA
cd /QuestionAnswering
allennlp train [config file] -s [training_directory] --include-package src
```

### Cite

If you use the code, please cite this paper:

```
@inproceedings{yao2022prompt,
    title = {Prompt Tuning for Discriminative Pre-trained Language Models},
    author = {Yuan, Yao and Bowen, Dong and Ao, Zhang and Zhengyan, Zhang and Ruobing, Xie and Zhiyuan, Liu and Leyu, Lin and Maosong, Sun and Jianyong, Wang},
    booktitle = {Findings of ACL 2022},
    year = {2022},
}
```

