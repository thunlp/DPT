from transformers import AutoConfig, ElectraTokenizer, RobertaTokenizer, BertTokenizer
from textcls.datasets.processors import txtcls_output_modes, txtcls_tasks_num_labels
from textcls.models.model import *
import torch

# mapping table from model_name to corresponding model class
model_dict = {
    Bert.name: Bert,
    FSBertWithLabelMarker.name: FSBertWithLabelMarker,
    FSBertWithLabelInit.name: FSBertWithLabelInit,

    Electra.name: Electra,
    FSElectraWithLabelInit.name: FSElectraWithLabelInit,
    FSElectraWithLabelMarker.name: FSElectraWithLabelMarker,
    FSElectraWithLabelItself.name: FSElectraWithLabelItself,
    FSElectraAddLabelInit.name: FSElectraAddLabelInit,
    FSElectraAddLabelItself.name: FSElectraAddLabelItself,

    BeginWithLabels.name: BeginWithLabels,
    BeginWithLabelsAndDownline.name: BeginWithLabelsAndDownline,
    BeginWithSentiment.name: BeginWithSentiment,
    BeginWithSentimentAndDownline.name: BeginWithSentimentAndDownline,

    # FSBertWithMultiLabel.name: FSBertWithMultiLabel,
}


def build_model(model_args, data_args):
    try:
        num_labels = txtcls_tasks_num_labels[data_args.task_name]
    except KeyError:
        raise ValueError("Task not found: %s" % (data_args.task_name))

    config = AutoConfig.from_pretrained(
        model_args.config_name if model_args.config_name else model_base[model_args.model_name_or_path],
        num_labels=num_labels,
        finetuning_task=data_args.task_name,
        cache_dir=model_args.cache_dir,
    )
    if 'roberta' in model_args.model_name_or_path:
        tokenizer = RobertaTokenizer.from_pretrained(
            model_args.tokenizer_name if model_args.tokenizer_name else model_base[model_args.model_name_or_path],
            cache_dir=model_args.cache_dir,
        )
    elif 'bert' in model_args.model_name_or_path:
        tokenizer = BertTokenizer.from_pretrained(
            model_args.tokenizer_name if model_args.tokenizer_name else model_base[model_args.model_name_or_path],
            cache_dir=model_args.cache_dir,
        )
    else:
        tokenizer = ElectraTokenizer.from_pretrained(
            model_args.tokenizer_name if model_args.tokenizer_name else model_base[model_args.model_name_or_path],
            cache_dir=model_args.cache_dir,
        )
    print(tokenizer)
    
    model = model_dict[model_args.model_name_or_path](model_args, data_args, config, tokenizer)
    if model_args.model_pretrained_path:
        log = model.load_state_dict(torch.load(model_args.model_pretrained_path))
        print(log)
    return model