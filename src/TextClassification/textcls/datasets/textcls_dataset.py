import os
import random
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import List, Optional, Union, Callable, Dict

import torch
from torch.utils.data.dataset import Dataset

from filelock import FileLock

from transformers import PreTrainedTokenizerBase
from transformers import logging
from transformers.data.processors.utils import InputFeatures
from transformers import glue_convert_examples_to_features
from textcls.datasets.processors import txtcls_processors, txtcls_output_modes

logger = logging.get_logger(__name__)


@dataclass
class TxtClsDataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.

    Using `HfArgumentParser` we can turn this class
    into argparse arguments to be able to specify them on
    the command line.
    """

    task_name: str = field(metadata={"help": "The name of the task to train on: " + ", ".join(txtcls_processors.keys())})
    data_dir: str = field(
        metadata={"help": "The input data dir. Should contain the .tsv files (or other data files) for the task."}
    )
    data_cached_dir: str = field(
        default=None,
        metadata={"help": "The path to cache processed data."}
    )
    max_seq_length: int = field(
        default=128,
        metadata={
            "help": "The maximum total input sequence length after tokenization. Sequences longer "
            "than this will be truncated, sequences shorter will be padded."
        },
    )
    overwrite_cache: bool = field(
        default=False, metadata={"help": "Overwrite the cached training and evaluation sets"}
    )

    data_ratio: float = field(default=1.0, metadata={"help": "The ratio of each class."})
    data_num: int = field(default=None, metadata={"help": "Prior to data_num. The number of data each class."})

    random_seed: int = field(default=0, metadata={"help": "The seed for shuffle class"})

    def __post_init__(self):
        self.task_name = self.task_name.lower()


class Split(Enum):
    train = "train"
    dev = "dev"
    test = "test"


class TxtclsDataset(Dataset):
    """
    This will be superseded by a framework-agnostic approach
    soon.
    """

    args: TxtClsDataTrainingArguments
    output_mode: str
    features: List[Dict]
    # features: List[InputFeatures]

    def __init__(
        self,
        args: TxtClsDataTrainingArguments,
        tokenizer: Callable,
        limit_length: Optional[int] = None,
        mode: Union[str, Split] = Split.train,
        cache_dir: Optional[str] = None,
    ):
        self.args = args
        self.processor = txtcls_processors[args.task_name]()
        self.output_mode = txtcls_output_modes[args.task_name]
        if isinstance(mode, str):
            try:
                mode = Split[mode]
            except KeyError:
                raise KeyError("mode is not a valid split name")
        # Load data features from cache or dataset file
        cached_features_file = os.path.join(
            cache_dir if cache_dir is not None else args.data_dir,
            "cached_{}_{}_{}_{}".format(
                mode.value,
                tokenizer.__class__.__name__,
                str(args.max_seq_length),
                args.task_name,
            ),
        )
        label_list = self.processor.get_labels()
        label_texts = self.processor.get_label_texts()
        self.label_list = label_list
        self.label_texts = label_texts

        # Make sure only the first process in distributed training processes the dataset,
        # and the others will use the cache.
        lock_path = cached_features_file + ".lock"
        with FileLock(lock_path):
            if os.path.exists(cached_features_file) and not args.overwrite_cache:
                start = time.time()
                self.features = torch.load(cached_features_file)
                logger.info(
                    f"Loading features from cached file {cached_features_file} [took %.3f s]", time.time() - start
                )
            else:
                logger.info(f"Creating features from dataset file at {args.data_dir}")

                if mode == Split.dev:
                    examples = self.processor.get_dev_examples(args.data_dir)
                elif mode == Split.test:
                    examples = self.processor.get_test_examples(args.data_dir)
                else:
                    examples = self.processor.get_train_examples(args.data_dir)
                    examples = self._sample_with_cls_ratio(examples, self.args.data_ratio, data_num=self.args.data_num, random_seed=self.args.random_seed)
                if limit_length is not None:
                    examples = examples[:limit_length]
                self.features = txtcls_convert_examples_to_features(
                    examples,
                    tokenizer,
                    max_length=args.max_seq_length,
                    label_list=label_list,
                    label_texts=label_texts,
                    output_mode=self.output_mode,
                )
                start = time.time()
                torch.save(self.features, cached_features_file)
                # ^ This seems to take a lot of time so I want to investigate why and how we can improve.
                logger.info(
                    "Saving features into cached file %s [took %.3f s]", cached_features_file, time.time() - start
                )

    def _sample_with_cls_ratio(self, examples, ratio, data_num=None, random_seed=None):
        if 1 <= ratio < 2:
            return examples
        if ratio >= 2 and data_num is None:
            data_num = int(ratio)

        if random_seed is not None:
            random.seed(random_seed)

        dic = {}
        # construct sample dic for each class
        for i, sample in enumerate(examples):
            lb = sample.label
            if lb not in dic:
                dic[lb] = []
            dic[lb].append(i)
        for k, v in dic.items():
            random.shuffle(v)
        # construct sampled data index list
        l = []
        for k, v in dic.items():
            v_len = len(v)
            n = data_num if data_num is not None else int(v_len*ratio)
            v = v[: n]
            l.extend(v)
        l = sorted(l)

        # sample data in examples
        examples = [examples[i] for i in l]
        return examples

    def __len__(self):
        return len(self.features)

    def __getitem__(self, i) -> InputFeatures:
        return self.features[i]

    def get_labels(self):
        return self.label_list


def txtcls_convert_examples_to_features(
        examples,
        tokenizer,
        max_length,
        task=None,
        label_list=None,
        label_texts=None,
        output_mode=None,
):
    if max_length is None:
        max_length = tokenizer.max_len

    label_map = {label: i for i, label in enumerate(label_list)}

    def label_from_example(example) -> Union[int, float, None]:
        if example.label is None:
            return None
        if output_mode == "classification":
            return label_map[example.label]
        elif output_mode == "regression":
            return float(example.label)
        raise KeyError(output_mode)

    labels = [label_from_example(example) for example in examples]

    batch_encoding = tokenizer(
        [(example.text_a, example.text_b) for example in examples],
        max_length=max_length,
        padding="max_length",
        truncation=True,
        labels=label_texts,
    )

    features = []
    for i in range(len(examples)):
        inputs = {k: batch_encoding[k][i] for k in batch_encoding}
        inputs['label'] = labels[i]
        # feature = InputFeatures(**inputs, label=labels[i])
        feature = inputs
        features.append(feature)

    return features
    # return glue_convert_examples_to_features(examples, tokenizer, max_length, task, label_list, output_mode)