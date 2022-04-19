import math
import os
import warnings
from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.utils.checkpoint
from torch import nn
from torch.nn import CrossEntropyLoss, MSELoss, Parameter
from transformers import BertModel, BertPreTrainedModel, ElectraModel, ElectraForPreTraining, ElectraPreTrainedModel
from textcls.datasets.processors import txtcls_processors

from textcls.models.bert_model import model_base
from textcls.models.bert_model import Bert, FSBertWithLabelMarker, FSBertWithLabelInit
from textcls.models.electra_model import Electra, FSElectraWithLabelMarker, FSElectraWithLabelItself, FSElectraWithLabelInit, FSElectraAddLabelItself, FSElectraAddLabelInit
from textcls.models.prompt_model import BeginWithLabels, BeginWithLabelsAndDownline, BeginWithSentiment, BeginWithSentimentAndDownline
