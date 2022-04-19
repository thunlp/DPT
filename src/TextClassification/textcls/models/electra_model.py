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
from textcls.models.tool import tokenize_all_data_with_labels, model_base

class Electra(ElectraPreTrainedModel):
    name: str = "electra"

    def __init__(self, model_args, data_args, config, tokenizer):
        super().__init__(config)
        self.num_labels = config.num_labels

        self.electra = ElectraModel.from_pretrained(
            model_base[model_args.model_name_or_path],
            config=config,
            cache_dir=model_args.cache_dir
        )

        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)

        self.tokenizer = tokenizer
        self.labels = None

    def tokenize(self, data, **kwargs):
        """
        :param data: List[(sentence A, sentence B)]
        :param kwargs:
        :return:
        """
        kwargs.pop("labels")
        dic = self.tokenizer(data, **kwargs)
        return dic

    def forward(
            self,
            input_ids=None,
            attention_mask=None,
            token_type_ids=None,
            position_ids=None,
            head_mask=None,
            inputs_embeds=None,
            labels=None,
            output_attentions=None,
            output_hidden_states=None,
            return_dict=None,
            **kwargs,
    ):
        r"""
        labels (:obj:`torch.LongTensor` of shape :obj:`(batch_size,)`, `optional`):
            Labels for computing the sequence classification/regression loss.
            Indices should be in :obj:`[0, ..., config.num_labels - 1]`.
            If :obj:`config.num_labels == 1` a regression loss is computed (Mean-Square loss),
            If :obj:`config.num_labels > 1` a classification loss is computed (Cross-Entropy).
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.electra(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        pooled_output = outputs[0][:, 0, :]

        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)

        loss = None
        if labels is not None:
            if self.num_labels == 1:
                #  We are doing regression
                loss_fct = MSELoss()
                loss = loss_fct(logits.view(-1), labels.view(-1))
            else:
                loss_fct = CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))

        output = (logits,) + outputs[2:]
        return ((loss,) + output) if loss is not None else output


class FSElectraWithLabelMarker(ElectraPreTrainedModel):
    name: str = "fs_electra_with_label_marker"

    def __init__(self, model_args, data_args, config, tokenizer):
        super().__init__(config)
        self.num_labels = config.num_labels

        self.electra = ElectraForPreTraining.from_pretrained(
            model_base[model_args.model_name_or_path],
            config=config,
            cache_dir=model_args.cache_dir
        )

        self.tokenizer = tokenizer
        self.labels = txtcls_processors[data_args.task_name].get_label_texts()
        self.loss_func = model_args.loss_func
        self.bias = Parameter(torch.zeros(self.num_labels))


        # init classification tokens with label embeddings
        label_tokens_ids = [self.tokenizer.convert_tokens_to_ids(self.tokenizer.tokenize(lb)) for i, lb in enumerate(self.labels)]

        # implementation 1
        self.label_postions = []
        self.cls_tokens = []
        flat_label_tokens_ids = []
        i = 0
        print(label_tokens_ids)
        for token_idxs in label_tokens_ids:
            if type(token_idxs) is not list:
                token_idxs = [token_idxs]
            self.label_postions.append(i)
            for tid in token_idxs:
                self.cls_tokens.append("[unused{}]".format(i + 100))
                flat_label_tokens_ids.append(tid)
                i += 1
        cls_token_ids = self.tokenizer.convert_tokens_to_ids(self.cls_tokens)
        # for cls_id, label_id in zip(cls_token_ids, flat_label_tokens_ids):
        #     with torch.no_grad():
        #         val = self.electra.electra.embeddings.word_embeddings.weight.data[label_id]
        #         self.electra.electra.embeddings.word_embeddings.weight[cls_id].copy_(val)
        print(self.label_postions, self.cls_tokens, self.tokenizer.convert_ids_to_tokens(flat_label_tokens_ids))

    def tokenize(self, data, **kwargs):
        """
         :param data: List[(sentence A, sentence B)]
         :param kwargs:
         :return:
         """
        max_len = kwargs.pop("max_length", 512)
        padding = kwargs.pop("padding", None)
        truncation = kwargs.pop("truncation", False)

        # construct label tokens
        label_tokens = self.cls_tokens
        # calculate label_positions
        # label_positions = [i for i in range(len(label_tokens))]
        label_positions = self.label_postions
        label_tokens = label_tokens + ["[SEP]"]

        dic = tokenize_all_data_with_labels(data, self.tokenizer, label_tokens, max_len, label_positions)
        return dic

    def forward(
            self,
            input_ids=None,
            attention_mask=None,
            token_type_ids=None,
            position_ids=None,
            head_mask=None,
            inputs_embeds=None,
            labels=None,
            output_attentions=None,
            output_hidden_states=None,
            return_dict=None,
            positions=None,
            **kwargs,
    ):
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.electra(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        raw_logits = outputs[0]
        cls_logits = []
        if len(raw_logits.size()) == 1:
            raw_logits = raw_logits.unsqueeze(0)
        batch_size = raw_logits.size(0)
        for i in range(self.num_labels):
            # torch.Size(batch_size, token_len)
            # print(raw_logits.size(), positions[:, i].size(), batch_size)
            i_cls_logits = raw_logits[torch.arange(batch_size), positions[:, i]]
            cls_logits.append(i_cls_logits.unsqueeze(1))
        logits = torch.cat(cls_logits, 1) + self.bias

        loss = None
        if labels is not None:
            targets = torch.zeros_like(logits.view(-1, self.num_labels), device=logits.device).scatter_(1, labels.view(-1, 1),1)
            loss_fct = nn.BCELoss()
            logits = 1 - torch.sigmoid(logits)
            loss = loss_fct(logits.view(-1, self.num_labels), targets)
        output = (logits,) + outputs[2:]
        return ((loss,) + output) if loss is not None else output


class FSElectraWithLabelItself(ElectraPreTrainedModel):
    name: str = "fs_electra_with_label_itself"

    def __init__(self, model_args, data_args, config, tokenizer):
        super().__init__(config)
        self.num_labels = config.num_labels

        self.electra = ElectraForPreTraining.from_pretrained(
            model_base[model_args.model_name_or_path],
            config=config,
            cache_dir=model_args.cache_dir
        )

        self.tokenizer = tokenizer
        self.labels = txtcls_processors[data_args.task_name].get_label_texts()
        self.loss_func = model_args.loss_func
        self.bias = Parameter(torch.zeros(self.num_labels))


        # init classification tokens with label embeddings
        label_tokens_ids = [self.tokenizer.convert_tokens_to_ids(self.tokenizer.tokenize(lb)) for i, lb in enumerate(self.labels)]

        # implementation 1
        self.label_postions = []
        self.cls_tokens = []
        flat_label_tokens_ids = []
        i = 0
        print(label_tokens_ids)
        for token_idxs in label_tokens_ids:
            if type(token_idxs) is not list:
                token_idxs = [token_idxs]
            self.label_postions.append(i)
            for tid in token_idxs:
                flat_label_tokens_ids.append(tid)
                # self.cls_tokens.append("[unused{}]".format(i + 100))
                self.cls_tokens.append(tid)
                i += 1
        cls_token_ids = self.tokenizer.convert_tokens_to_ids(self.cls_tokens)
        # for cls_id, label_id in zip(cls_token_ids, flat_label_tokens_ids):
        #     with torch.no_grad():
        #         val = self.electra.electra.embeddings.word_embeddings.weight.data[label_id]
        #         self.electra.electra.embeddings.word_embeddings.weight[cls_id].copy_(val)
        print(self.label_postions, self.cls_tokens, self.tokenizer.convert_ids_to_tokens(flat_label_tokens_ids))

    def tokenize(self, data, **kwargs):
        """
         :param data: List[(sentence A, sentence B)]
         :param kwargs:
         :return:
         """
        max_len = kwargs.pop("max_length", 512)
        padding = kwargs.pop("padding", None)
        truncation = kwargs.pop("truncation", False)

        # construct label tokens
        label_tokens = self.cls_tokens
        # calculate label_positions
        # label_positions = [i for i in range(len(label_tokens))]
        label_positions = self.label_postions
        label_tokens = label_tokens + ["[SEP]"]

        dic = tokenize_all_data_with_labels(data, self.tokenizer, label_tokens, max_len, label_positions)
        return dic

    def forward(
            self,
            input_ids=None,
            attention_mask=None,
            token_type_ids=None,
            position_ids=None,
            head_mask=None,
            inputs_embeds=None,
            labels=None,
            output_attentions=None,
            output_hidden_states=None,
            return_dict=None,
            positions=None,
            **kwargs,
    ):
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.electra(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        raw_logits = outputs[0]
        cls_logits = []
        if len(raw_logits.size()) == 1:
            raw_logits = raw_logits.unsqueeze(0)
        batch_size = raw_logits.size(0)
        for i in range(self.num_labels):
            # torch.Size(batch_size, token_len)
            # print(raw_logits.size(), positions[:, i].size(), batch_size)
            i_cls_logits = raw_logits[torch.arange(batch_size), positions[:, i]]
            cls_logits.append(i_cls_logits.unsqueeze(1))
        logits = torch.cat(cls_logits, 1) + self.bias

        loss = None
        if labels is not None:
            targets = torch.zeros_like(logits.view(-1, self.num_labels), device=logits.device).scatter_(1, labels.view(-1, 1),1)
            loss_fct = nn.BCELoss()
            logits = 1 - torch.sigmoid(logits)
            loss = loss_fct(logits.view(-1, self.num_labels), targets)
        output = (logits,) + outputs[2:]
        return ((loss,) + output) if loss is not None else output


class FSElectraAddLabelItself(ElectraPreTrainedModel):
    name: str = "fs_electra_add_label_itself"

    def __init__(self, model_args, data_args, config, tokenizer):
        super().__init__(config)
        self.num_labels = config.num_labels

        self.electra = ElectraForPreTraining.from_pretrained(
            model_base[model_args.model_name_or_path],
            config=config,
            cache_dir=model_args.cache_dir
        )

        self.tokenizer = tokenizer
        self.labels = txtcls_processors[data_args.task_name].get_label_texts()
        self.loss_func = model_args.loss_func
        self.bias = Parameter(torch.zeros(self.num_labels))


        # init classification tokens with label embeddings
        label_tokens_ids = [self.tokenizer.convert_tokens_to_ids(["[unused{}]".format(i + 200)] + self.tokenizer.tokenize(lb)) for i, lb in enumerate(self.labels)]

        # implementation 1
        self.label_postions = []
        self.cls_tokens = []
        flat_label_tokens_ids = []
        i = 0
        print(label_tokens_ids)
        for token_idxs in label_tokens_ids:
            if type(token_idxs) is not list:
                token_idxs = [token_idxs]
            self.label_postions.append(i)
            for tid in token_idxs:
                flat_label_tokens_ids.append(tid)
                # self.cls_tokens.append("[unused{}]".format(i + 100))
                self.cls_tokens.append(tid)
                i += 1
        cls_token_ids = self.tokenizer.convert_tokens_to_ids(self.cls_tokens)
        # for cls_id, label_id in zip(cls_token_ids, flat_label_tokens_ids):
        #     with torch.no_grad():
        #         val = self.electra.electra.embeddings.word_embeddings.weight.data[label_id]
        #         self.electra.electra.embeddings.word_embeddings.weight[cls_id].copy_(val)
        print(self.label_postions, self.cls_tokens, self.tokenizer.convert_ids_to_tokens(flat_label_tokens_ids))

    def tokenize(self, data, **kwargs):
        """
         :param data: List[(sentence A, sentence B)]
         :param kwargs:
         :return:
         """
        max_len = kwargs.pop("max_length", 512)
        padding = kwargs.pop("padding", None)
        truncation = kwargs.pop("truncation", False)

        # construct label tokens
        label_tokens = self.cls_tokens
        # calculate label_positions
        # label_positions = [i for i in range(len(label_tokens))]
        label_positions = self.label_postions
        label_tokens = label_tokens + ["[SEP]"]

        dic = tokenize_all_data_with_labels(data, self.tokenizer, label_tokens, max_len, label_positions)
        return dic

    def forward(
            self,
            input_ids=None,
            attention_mask=None,
            token_type_ids=None,
            position_ids=None,
            head_mask=None,
            inputs_embeds=None,
            labels=None,
            output_attentions=None,
            output_hidden_states=None,
            return_dict=None,
            positions=None,
            **kwargs,
    ):
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.electra(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        raw_logits = outputs[0]
        cls_logits = []
        if len(raw_logits.size()) == 1:
            raw_logits = raw_logits.unsqueeze(0)
        batch_size = raw_logits.size(0)
        for i in range(self.num_labels):
            # torch.Size(batch_size, token_len)
            # print(raw_logits.size(), positions[:, i].size(), batch_size)
            i_cls_logits = raw_logits[torch.arange(batch_size), positions[:, i]]
            cls_logits.append(i_cls_logits.unsqueeze(1))
        logits = torch.cat(cls_logits, 1) + self.bias

        loss = None
        if labels is not None:
            targets = torch.zeros_like(logits.view(-1, self.num_labels), device=logits.device).scatter_(1, labels.view(-1, 1),1)
            loss_fct = nn.BCELoss()
            logits = 1 - torch.sigmoid(logits)
            loss = loss_fct(logits.view(-1, self.num_labels), targets)
        output = (logits,) + outputs[2:]
        return ((loss,) + output) if loss is not None else output


class FSElectraWithLabelInit(ElectraPreTrainedModel):
    name: str = "fs_electra_with_label_init"

    def __init__(self, model_args, data_args, config, tokenizer):
        super().__init__(config)
        self.num_labels = config.num_labels

        self.electra = ElectraForPreTraining.from_pretrained(
            model_base[model_args.model_name_or_path],
            config=config,
            cache_dir=model_args.cache_dir
        )

        self.tokenizer = tokenizer
        self.labels = txtcls_processors[data_args.task_name].get_label_texts()
        self.loss_func = model_args.loss_func
        self.bias = Parameter(torch.zeros(self.num_labels))


        # init classification tokens with label embeddings
        label_tokens_ids = [self.tokenizer.convert_tokens_to_ids(self.tokenizer.tokenize(lb)) for i, lb in enumerate(self.labels)]

        # implementation 1
        self.label_postions = []
        self.cls_tokens = []
        flat_label_tokens_ids = []
        i = 0
        print(label_tokens_ids)
        for token_idxs in label_tokens_ids:
            if type(token_idxs) is not list:
                token_idxs = [token_idxs]
            self.label_postions.append(i)
            for tid in token_idxs:
                self.cls_tokens.append("[unused{}]".format(i + 100))
                flat_label_tokens_ids.append(tid)
                i += 1
        cls_token_ids = self.tokenizer.convert_tokens_to_ids(self.cls_tokens)
        for cls_id, label_id in zip(cls_token_ids, flat_label_tokens_ids):
            with torch.no_grad():
                val = self.electra.electra.embeddings.word_embeddings.weight.data[label_id]
                self.electra.electra.embeddings.word_embeddings.weight[cls_id].copy_(val)
        print(self.label_postions, self.cls_tokens, self.tokenizer.convert_ids_to_tokens(flat_label_tokens_ids))

    def tokenize(self, data, **kwargs):
        """
         :param data: List[(sentence A, sentence B)]
         :param kwargs:
         :return:
         """
        max_len = kwargs.pop("max_length", 512)
        padding = kwargs.pop("padding", None)
        truncation = kwargs.pop("truncation", False)

        # construct label tokens
        label_tokens = self.cls_tokens
        # calculate label_positions
        # label_positions = [i for i in range(len(label_tokens))]
        label_positions = self.label_postions
        label_tokens = label_tokens + ["[SEP]"]

        dic = tokenize_all_data_with_labels(data, self.tokenizer, label_tokens, max_len, label_positions)
        return dic

    def forward(
            self,
            input_ids=None,
            attention_mask=None,
            token_type_ids=None,
            position_ids=None,
            head_mask=None,
            inputs_embeds=None,
            labels=None,
            output_attentions=None,
            output_hidden_states=None,
            return_dict=None,
            positions=None,
            **kwargs,
    ):
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.electra(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        raw_logits = outputs[0]
        cls_logits = []
        if len(raw_logits.size()) == 1:
            raw_logits = raw_logits.unsqueeze(0)
        batch_size = raw_logits.size(0)
        for i in range(self.num_labels):
            # torch.Size(batch_size, token_len)
            # print(raw_logits.size(), positions[:, i].size(), batch_size)
            i_cls_logits = raw_logits[torch.arange(batch_size), positions[:, i]]
            cls_logits.append(i_cls_logits.unsqueeze(1))
        logits = torch.cat(cls_logits, 1) + self.bias

        loss = None
        if labels is not None:
            targets = torch.zeros_like(logits.view(-1, self.num_labels), device=logits.device).scatter_(1, labels.view(-1, 1),1)
            loss_fct = nn.BCELoss()
            logits = 1 - torch.sigmoid(logits)
            loss = loss_fct(logits.view(-1, self.num_labels), targets)
        output = (logits,) + outputs[2:]
        return ((loss,) + output) if loss is not None else output


class FSElectraAddLabelInit(ElectraPreTrainedModel):
    name: str = "fs_electra_add_label_init"

    def __init__(self, model_args, data_args, config, tokenizer):
        super().__init__(config)
        self.num_labels = config.num_labels

        self.electra = ElectraForPreTraining.from_pretrained(
            model_base[model_args.model_name_or_path],
            config=config,
            cache_dir=model_args.cache_dir
        )

        self.tokenizer = tokenizer
        self.labels = txtcls_processors[data_args.task_name].get_label_texts()
        self.loss_func = model_args.loss_func
        self.bias = Parameter(torch.zeros(self.num_labels))


        # init classification tokens with label embeddings
        label_tokens_ids = [self.tokenizer.convert_tokens_to_ids(["[unused{}]".format(i + 200)] + self.tokenizer.tokenize(lb)) for i, lb in enumerate(self.labels)]

        # implementation 1
        self.label_postions = []
        self.cls_tokens = []
        flat_label_tokens_ids = []
        i = 0
        print(label_tokens_ids)
        for token_idxs in label_tokens_ids:
            if type(token_idxs) is not list:
                token_idxs = [token_idxs]
            self.label_postions.append(i)
            for tid in token_idxs:
                self.cls_tokens.append("[unused{}]".format(i + 100))
                flat_label_tokens_ids.append(tid)
                i += 1
        cls_token_ids = self.tokenizer.convert_tokens_to_ids(self.cls_tokens)
        for cls_id, label_id in zip(cls_token_ids, flat_label_tokens_ids):
            with torch.no_grad():
                val = self.electra.electra.embeddings.word_embeddings.weight.data[label_id]
                self.electra.electra.embeddings.word_embeddings.weight[cls_id].copy_(val)
        print(self.label_postions, self.cls_tokens, self.tokenizer.convert_ids_to_tokens(flat_label_tokens_ids))

    def tokenize(self, data, **kwargs):
        """
         :param data: List[(sentence A, sentence B)]
         :param kwargs:
         :return:
         """
        max_len = kwargs.pop("max_length", 512)
        padding = kwargs.pop("padding", None)
        truncation = kwargs.pop("truncation", False)

        # construct label tokens
        label_tokens = self.cls_tokens
        # calculate label_positions
        # label_positions = [i for i in range(len(label_tokens))]
        label_positions = self.label_postions
        label_tokens = label_tokens + ["[SEP]"]

        dic = tokenize_all_data_with_labels(data, self.tokenizer, label_tokens, max_len, label_positions)
        return dic

    def forward(
            self,
            input_ids=None,
            attention_mask=None,
            token_type_ids=None,
            position_ids=None,
            head_mask=None,
            inputs_embeds=None,
            labels=None,
            output_attentions=None,
            output_hidden_states=None,
            return_dict=None,
            positions=None,
            **kwargs,
    ):
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.electra(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        raw_logits = outputs[0]
        cls_logits = []
        if len(raw_logits.size()) == 1:
            raw_logits = raw_logits.unsqueeze(0)
        batch_size = raw_logits.size(0)
        for i in range(self.num_labels):
            # torch.Size(batch_size, token_len)
            # print(raw_logits.size(), positions[:, i].size(), batch_size)
            i_cls_logits = raw_logits[torch.arange(batch_size), positions[:, i]]
            cls_logits.append(i_cls_logits.unsqueeze(1))
        logits = torch.cat(cls_logits, 1) + self.bias

        loss = None
        if labels is not None:
            targets = torch.zeros_like(logits.view(-1, self.num_labels), device=logits.device).scatter_(1, labels.view(-1, 1),1)
            loss_fct = nn.BCELoss()
            logits = 1 - torch.sigmoid(logits)
            loss = loss_fct(logits.view(-1, self.num_labels), targets)
        output = (logits,) + outputs[2:]
        return ((loss,) + output) if loss is not None else output