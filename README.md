### Computing infrastructure

#### OS:

Distributor ID: Ubuntu
Description:    Ubuntu 16.04.1 LTS
Release:    16.04
Codename:   xenial

#### GPU:

GeForce RTX 2080Ti

#### Language:

Python 3.7.5

#### Required packages:

For Text Classification, can install with:

```
cd src/TextClassification/ 
pip install -r requirements.txt
```

For Question Answering, can build by docker with:

```
cd src/QuestionAnswering/
docker build -t dpt:v0 .
docker run --gpus '"device=0,1,2,3,4,5"' -it -v [your path]/src/QuestionAnswering:/QuestionAnswering  --name=DPT_QA dpt:v0
```

### Hyperparameters:

The method of choosing hyperparameter values: Grid Search.

The detailed description can be found in the appendix of the paper.


### Run:

#### Text Classification

```
cd src/TextClassification/
bash run_textclassification.sh
```

#### Question Answering

```
docker attach DPT_QA
cd /QuestionAnswering
allennlp train [config file] -s [training_directory] --include-package src
```
