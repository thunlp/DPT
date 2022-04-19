from typing import List, Tuple, Dict
import html
import re
import sys
import os
import pickle
import unicodedata

from spacy.tokenizer import Tokenizer as STokenizer
from spacy.lang.en import English

# from allennlp.data.tokenizers import Token
# from allennlp.data.tokenizers import SpacyTokenizer
# from src.data.dataset_readers.spacy_tokenizer.py import SpacyTokenizer
# from allennlp.data.dataset_readers.reading_comprehension.drop import DropReader
# from allennlp.data.dataset_readers.reading_comprehension.util import split_tokens_by_hyphen
import string

from typing import List, Optional

from overrides import overrides
import spacy
from spacy.tokens import Doc

# from allennlp.common.util import get_spacy_model
# from allennlp.data.tokenizers.token import Token
from allennlp.data.tokenizers.tokenizer import Tokenizer 

from typing import Any, Callable, Dict, List, Tuple, TypeVar, Iterable, Iterator

import spacy
from spacy.cli.download import download as spacy_download
from spacy.language import Language as SpacyModelType

# import logger
from typing import NamedTuple

class Token(NamedTuple):

    text: str = None
    idx: int = None
    lemma_: str = None
    pos_: str = None
    tag_: str = None
    dep_: str = None
    ent_type_: str = None
    text_id: int = None
    type_id: int = None

    def __str__(self):
        return self.text

    def __repr__(self):
        return self.__str__()


def show_token(token: Token) -> str:
    return (
        f"{token.text} "
        f"(idx: {token.idx}) "
        f"(lemma: {token.lemma_}) "
        f"(pos: {token.pos_}) "
        f"(tag: {token.tag_}) "
        f"(dep: {token.dep_}) "
        f"(ent_type: {token.ent_type_}) "
        f"(text_id: {token.text_id}) "
        f"(type_id: {token.type_id}) "
    )




LOADED_SPACY_MODELS: Dict[Tuple[str, bool, bool, bool], SpacyModelType] = {}


def get_spacy_model(
    spacy_model_name: str, pos_tags: bool, parse: bool, ner: bool
) -> SpacyModelType:
    """
    In order to avoid loading spacy models a whole bunch of times, we'll save references to them,
    keyed by the options we used to create the spacy model, so any particular configuration only
    gets loaded once.
    """

    options = (spacy_model_name, pos_tags, parse, ner)
    if options not in LOADED_SPACY_MODELS:
        disable = ["vectors", "textcat"]
        if not pos_tags:
            disable.append("tagger")
        if not parse:
            disable.append("parser")
        if not ner:
            disable.append("ner")
        try:
            spacy_model = spacy.load(spacy_model_name, disable=disable)
        except OSError:
            # logger.warning(
            #     f"Spacy models '{spacy_model_name}' not found.  Downloading and installing."
            # )
            print ('downloading')
            spacy_download(spacy_model_name)
            # NOTE(mattg): The following four lines are a workaround suggested by Ines for spacy
            # 2.1.0, which removed the linking that was done in spacy 2.0.  importlib doesn't find
            # packages that were installed in the same python session, so the way `spacy_download`
            # works in 2.1.0 is broken for this use case.  These four lines can probably be removed
            # at some point in the future, once spacy has figured out a better way to handle this.
            # See https://github.com/explosion/spaCy/issues/3435.
            from spacy.cli import link
            from spacy.util import get_package_path

            package_path = get_package_path(spacy_model_name)
            link(spacy_model_name, spacy_model_name, model_path=package_path)
            spacy_model = spacy.load(spacy_model_name, disable=disable)

        LOADED_SPACY_MODELS[options] = spacy_model
    return LOADED_SPACY_MODELS[options]

@Tokenizer.register("spacy")
class SpacyTokenizer(Tokenizer):
    """
    A ``Tokenizer`` that uses spaCy's tokenizer.  It's fast and reasonable - this is the
    recommended ``Tokenizer``. By default it will return allennlp Tokens,
    which are small, efficient NamedTuples (and are serializable). If you want
    to keep the original spaCy tokens, pass keep_spacy_tokens=True.  Note that we leave one particular piece of
    post-processing for later: the decision of whether or not to lowercase the token.  This is for
    two reasons: (1) if you want to make two different casing decisions for whatever reason, you
    won't have to run the tokenizer twice, and more importantly (2) if you want to lowercase words
    for your word embedding, but retain capitalization in a character-level representation, we need
    to retain the capitalization here.

    Parameters
    ----------
    language : ``str``, optional, (default="en_core_web_sm")
        Spacy model name.
    pos_tags : ``bool``, optional, (default=False)
        If ``True``, performs POS tagging with spacy model on the tokens.
        Generally used in conjunction with :class:`~allennlp.data.token_indexers.pos_tag_indexer.PosTagIndexer`.
    parse : ``bool``, optional, (default=False)
        If ``True``, performs dependency parsing with spacy model on the tokens.
        Generally used in conjunction with :class:`~allennlp.data.token_indexers.pos_tag_indexer.DepLabelIndexer`.
    ner : ``bool``, optional, (default=False)
        If ``True``, performs dependency parsing with spacy model on the tokens.
        Generally used in conjunction with :class:`~allennlp.data.token_indexers.ner_tag_indexer.NerTagIndexer`.
    keep_spacy_tokens : ``bool``, optional, (default=False)
        If ``True``, will preserve spacy token objects, We copy spacy tokens into our own class by default instead
        because spacy Cython Tokens can't be pickled.
    split_on_spaces : ``bool``, optional, (default=False)
        If ``True``, will split by spaces without performing tokenization.
        Used when your data is already tokenized, but you want to perform pos, ner or parsing on the tokens.
    start_tokens : ``Optional[List[str]]``, optional, (default=None)
        If given, these tokens will be added to the beginning of every string we tokenize.
    end_tokens : ``Optional[List[str]]``, optional, (default=None)
        If given, these tokens will be added to the end of every string we tokenize.
    """

    def __init__(
        self,
        language: str = "en_core_web_sm",
        pos_tags: bool = False,
        parse: bool = False,
        ner: bool = False,
        keep_spacy_tokens: bool = False,
        split_on_spaces: bool = False,
        start_tokens: Optional[List[str]] = None,
        end_tokens: Optional[List[str]] = None,
    ) -> None:
        self.spacy = get_spacy_model(language, pos_tags, parse, ner)
        if split_on_spaces:
            self.spacy.tokenizer = _WhitespaceSpacyTokenizer(self.spacy.vocab)

        self._keep_spacy_tokens = keep_spacy_tokens
        self._start_tokens = start_tokens or []
        # We reverse the tokens here because we're going to insert them with `insert(0)` later;
        # this makes sure they show up in the right order.
        self._start_tokens.reverse()
        self._end_tokens = end_tokens or []

    def _sanitize(self, tokens: List[spacy.tokens.Token]) -> List[Token]:
        """
        Converts spaCy tokens to allennlp tokens. Is a no-op if
        keep_spacy_tokens is True
        """
        if not self._keep_spacy_tokens:
            tokens = [
                Token(
                    token.text,
                    token.idx,
                    token.lemma_,
                    token.pos_,
                    token.tag_,
                    token.dep_,
                    token.ent_type_,
                )
                for token in tokens
            ]
        for start_token in self._start_tokens:
            tokens.insert(0, Token(start_token, 0))
        for end_token in self._end_tokens:
            tokens.append(Token(end_token, -1))
        return tokens

    @overrides
    def batch_tokenize(self, texts: List[str]) -> List[List[Token]]:
        return [
            self._sanitize(_remove_spaces(tokens))
            for tokens in self.spacy.pipe(texts, n_threads=-1)
        ]

    @overrides
    def tokenize(self, text: str) -> List[Token]:
        # This works because our Token class matches spacy's.
        return self._sanitize(_remove_spaces(self.spacy(text)))


class _WhitespaceSpacyTokenizer:
    """
    Spacy doesn't assume that text is tokenised. Sometimes this
    is annoying, like when you have gold data which is pre-tokenised,
    but Spacy's tokenisation doesn't match the gold. This can be used
    as follows:
    nlp = spacy.load("en_core_web_md")
    # hack to replace tokenizer with a whitespace tokenizer
    nlp.tokenizer = _WhitespaceSpacyTokenizer(nlp.vocab)
    ... use nlp("here is some text") as normal.
    """

    def __init__(self, vocab):
        self.vocab = vocab

    def __call__(self, text):
        words = text.split(" ")
        spaces = [True] * len(words)
        return Doc(self.vocab, words=words, spaces=spaces)


def _remove_spaces(tokens: List[spacy.tokens.Token]) -> List[spacy.tokens.Token]:
    return [token for token in tokens if not token.is_space]


STRIPPED_CHARACTERS = string.punctuation + "".join([u"‘", u"’", u"´", u"`", "_"])


whitespaces = re.findall(r'\s', u''.join(chr(c) for c in range(sys.maxunicode+1)), re.UNICODE)
empty_chars = ['\u200b', '\ufeff', '\u2061'] # zero width space, byte order mark
def standardize_text_simple(text, deletions_tracking=False):
    for whitespace in whitespaces:
        text = text.replace(whitespace, ' ')

    if deletions_tracking:
        deletion_indexes = {}
        track_deletions(text, deletion_indexes)

    # This is a must for proper tokenization with offsets
    for empty_char in empty_chars:
        text = text.replace(empty_char, '')

    text = ' '.join(text.split()) # use ' ' for all spaces and replace sequence of spaces with single space

    return text if not deletions_tracking else (text, deletion_indexes)

def track_deletions(text, deletion_indexes):
    """
    Track deletions for empty_chars removal and space sequences trimming
    """
    for empty_char in empty_chars:
        for i, char in enumerate(text):
            if char == empty_char:
                deletion_indexes[i] = 1

    initial_space_length = len(text) - len(text.lstrip())
    if initial_space_length > 0:
        deletion_indexes[0] = initial_space_length
    space_sequence = False
    length = 0
    for i, char in enumerate(text):
        if char == ' ':
            space_sequence = True
            length += 1
            if i == len(text) - 1:
                deletion_indexes[i - length] = length
        else:
            if space_sequence:
                if length > 1 and (i - length) > 0:
                    deletion_indexes[i - length] = length - 1
            space_sequence = False
            length = 0

def standardize_text_advanced(text, deletions_tracking=False):
    text = html.unescape(text)
    text = standardize_text_simple(text)

    # There is a pattern that repeats itself 97 times in the train set and 16 in the
    # dev set: "<letters>.:<digits>". It originates from the method of parsing the
    # Wikipedia pages. In such an occurrence, "<letters>." is the last word of a
    # sentence, followed by a period. Then, in the wikipedia page, follows a superscript
    # of digits within square brackets, which is a hyperlink to a reference. After the
    # hyperlink there is a colon, ":", followed by <digits>. These digits are the page
    # within the reference.
    # Example: https://en.wikipedia.org/wiki/Polish%E2%80%93Ottoman_War_(1672%E2%80%931676)
    if '.:' in text:
        text = re.sub('\.:\d+(-\d+)*', '.', text)

    # In a few cases the passage starts with a location and weather description. 
    # Example: "at Lincoln Financial Field, Philadelphia|weather= 60&#160;&#176;F (Clear)".
    # Relevant for 3 passages (15 questions) in the training set and 1 passage (25 questions) in the dev set.
    text.replace("|weather", " weather")

    return text if not deletions_tracking else (text, {})

def custom_word_tokenizer():
    #tokenizer_exceptions = English().Defaults.tokenizer_exceptions
    word_tokenizer = SpacyTokenizer()
    word_tokenizer.spacy.tokenizer = STokenizer(vocab=word_tokenizer.spacy.tokenizer.vocab, rules={}, 
                                               prefix_search=word_tokenizer.spacy.tokenizer.prefix_search, 
                                               suffix_search=word_tokenizer.spacy.tokenizer.suffix_search, 
                                               infix_finditer=word_tokenizer.spacy.tokenizer.infix_finditer)


    return word_tokenizer


def split_token_by_delimiter(token: Token, delimiter: str) -> List[Token]:
    """
    From allennlp's reading_comprehension.util
    """
    split_tokens = []
    char_offset = token.idx
    for sub_str in token.text.split(delimiter):
        if sub_str:
            split_tokens.append(Token(text=sub_str, idx=char_offset))
            char_offset += len(sub_str)
        split_tokens.append(Token(text=delimiter, idx=char_offset))
        char_offset += len(delimiter)
    if split_tokens:
        split_tokens.pop(-1)
        char_offset -= len(delimiter)
        return split_tokens
    else:
        return [token]

def split_tokens_by_hyphen(tokens: List[Token]) -> List[Token]:
    """
    From allennlp's reading_comprehension.util
    """
    hyphens = ["-", "–", "~", "—"]
    new_tokens: List[Token] = []

    for token in tokens:
        if any(hyphen in token.text for hyphen in hyphens):
            unsplit_tokens = [token]
            split_tokens: List[Token] = []
            for hyphen in hyphens:
                for unsplit_token in unsplit_tokens:
                    if hyphen in token.text:
                        split_tokens += split_token_by_delimiter(unsplit_token, hyphen)
                    else:
                        split_tokens.append(unsplit_token)
                unsplit_tokens, split_tokens = split_tokens, []
            new_tokens += unsplit_tokens
        else:
            new_tokens.append(token)

    return new_tokens

def run_strip_accents(text):
    """
    From tokenization_bert.py by huggingface/transformers.
    Strips accents from a piece of text.
    """
    text = unicodedata.normalize("NFD", text)
    output = []
    for char in text:
        cat = unicodedata.category(char)
        if cat == "Mn":
            continue
        output.append(char)
    return "".join(output)

def find_all(substr, text):
    matches = []
    start = 0
    while True:
        start = text.find(substr, start)
        if start == -1:
            break
        matches.append(start)
        start += 1
    return matches

def index_text_to_tokens(text, tokens):
    text_index_to_token_index = []
    token_index = 0
    next_token_index = token_index + 1
    index = tokens[token_index].idx
    next_index = tokens[next_token_index].idx
    for i in range(len(text)):
        while True:
            while next_index == index and next_token_index < len(tokens) - 1:
                next_token_index += 1
                next_index = tokens[next_token_index].idx
            if next_index == index and next_token_index == len(tokens) - 1:
                next_token_index += 1
                next_index = len(text)
            
            if i >= index and i < next_index:
                text_index_to_token_index.append(token_index)
                break
            else:
                token_index = next_token_index
                index = next_index

                if next_token_index < len(tokens) - 1:
                    next_token_index += 1
                    next_index = tokens[next_token_index].idx
                else:
                    next_token_index += 1
                    next_index = len(text)
                if (next_token_index > len(tokens)):
                    raise Exception("Error in " + text)
    return text_index_to_token_index

def find_valid_spans(text: str, answer_texts: List[str], 
                     text_index_to_token_index: List[int], 
                     tokens: List[Token], wordpieces: List[List[int]],
                     gold_indexes: List[int]) -> List[Tuple[int, int]]:
    text = text.lower()
    answer_texts_set = set()
    for answer_text in answer_texts:
        option1 = answer_text.lower()
        option2 = option1.strip(STRIPPED_CHARACTERS) 
        option3 = run_strip_accents(option1)
        option4 = run_strip_accents(option2)
        answer_texts_set.update([option1, option2, option3, option4])

    gold_token_indexes = None
    if gold_indexes is not None:
        gold_token_indexes = []
        for gold_index in gold_indexes:
            if gold_index < len(text_index_to_token_index): # if the gold index was not truncated
                if text[gold_index] == ' ' and gold_index < len(text_index_to_token_index) - 1:
                    gold_index += 1
                gold_token_indexes.append(text_index_to_token_index[gold_index])

    valid_spans = set()
    for answer_text in answer_texts_set:
        start_indexes = find_all(answer_text, text)
        for start_index in start_indexes:
            start_token_index = text_index_to_token_index[start_index]
            end_token_index = text_index_to_token_index[start_index + len(answer_text) - 1]

            wordpieces_condition = (wordpieces[start_token_index][0] == start_token_index and 
                                    wordpieces[end_token_index][-1] == end_token_index)

            stripped_answer_text = answer_text.strip(STRIPPED_CHARACTERS)
            stripped_first_token = tokens[start_token_index].lemma_.lower().strip(STRIPPED_CHARACTERS)
            stripped_last_token = tokens[end_token_index].lemma_.lower().strip(STRIPPED_CHARACTERS)
            text_match_condition = (stripped_answer_text.startswith(stripped_first_token) and 
                                        stripped_answer_text.endswith(stripped_last_token))

            gold_index_condition = gold_token_indexes is None or start_token_index in gold_token_indexes

            if wordpieces_condition and text_match_condition and gold_index_condition:
                valid_spans.add((start_token_index, end_token_index))

    return valid_spans

def save_pkl(instances, pickle_dict, is_training):
    if is_pickle_dict_valid(pickle_dict):
        pkl_path = get_pkl_path(pickle_dict, is_training)
        os.makedirs(os.path.dirname(pkl_path), exist_ok=True)
        with open(pkl_path, 'wb') as dataset_file:
            pickle.dump(instances, dataset_file)

def load_pkl(pickle_dict, is_training):
    try:
        with open(get_pkl_path(pickle_dict, is_training), 'rb') as dataset_pkl:
            return pickle.load(dataset_pkl)
    except Exception as e:
        return None

def get_pkl_path(pickle_dict, is_training):
    return os.path.join(pickle_dict['path'], f"{pickle_dict['file_name']}_{'train' if is_training else 'dev'}.pkl")

def is_pickle_dict_valid(pickle_dict):
    return pickle_dict is not None and 'path' in pickle_dict and 'file_name' in pickle_dict