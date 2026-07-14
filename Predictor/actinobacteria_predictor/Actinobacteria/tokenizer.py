"""Implements the local ProkBERT tokenizer used by the Actinobacteria predictor."""
import collections
import os
import json
from copy import deepcopy
from typing import List, Optional, Tuple, Dict, Set
from transformers import PreTrainedTokenizer
from transformers.utils import logging
from itertools import product
logger = logging.get_logger(__name__)


VOCAB_FILES_NAMES = {"vocab_file": "vocab.txt"}


PRETRAINED_VOCAB_FILES_MAP = {
    "vocab_file": {
        "lca-mini-k6s1": "lca-base-dna6/vocab.txt",
        "lca-mini-k6s2": "lca-base-dna6/vocab.txt",
        "lca-mini-k1s1": "lca-base-dna1/vocab.txt",
    }
}


PRETRAINED_POSITIONAL_EMBEDDINGS_SIZES = {
    "lca-mini-k6s1": 1024,
    "lca-mini-k1s1": 1024,
    "lca-mini-k6s2": 2048,
}


PRETRAINED_INIT_CONFIGURATION = {
    "lca-mini-k6s1": {"do_upper_case": True},
    "lca-mini-k1s1": {"do_upper_case": True},
    "lca-mini-k6s2": {"do_upper_case": True},
}

def generate_kmers(abc: Set[str], k: int) -> List[str]:


    return [''.join(p) for p in product(abc, repeat=k)]


def load_vocab(vocab_file):

    vocab = collections.OrderedDict()
    with open(vocab_file, "r", encoding="utf-8") as reader:
        tokens = reader.readlines()
    for index, token in enumerate(tokens):
        vocab[token.rstrip("\n")] = index
    return vocab


def resolve_vocab_file(vocab_file: Optional[str], kmer) -> str:


    if vocab_file and os.path.exists(vocab_file):
        return vocab_file


    try:
        import prokbert
        package_dir = os.path.dirname(prokbert.__file__)
        vocab_path = os.path.join(package_dir, 'data/prokbert_vocabs/', f'prokbert-base-dna{kmer}', 'vocab.txt')

        print(vocab_path)


        if os.path.exists(vocab_path):
            logger.info(f"Loaded vocab file from installed prokbert package: {vocab_path}")
            return vocab_path
    except ImportError:
        logger.info("Prokbert package not installed, proceeding to download vocab.txt.")


    github_url = "https://raw.githubusercontent.com/username/prokbert/main/vocab.txt"
    temp_vocab_path = os.path.join(os.getcwd(), "vocab.txt")
    try:
        import requests

        response = requests.get(github_url, timeout=10)
        response.raise_for_status()
        with open(temp_vocab_path, "w", encoding="utf-8") as f:
            f.write(response.text)
        logger.info(f"Downloaded vocab.txt from GitHub to: {temp_vocab_path}")
        return temp_vocab_path
    except requests.RequestException as e:
        raise FileNotFoundError(
            "Could not find or download vocab.txt. Ensure prokbert is installed or "
            "provide a valid vocab file path. Error: {e}"
        ) from e


class LCATokenizer(PreTrainedTokenizer):


    vocab_files_names = VOCAB_FILES_NAMES
    pretrained_vocab_files_map = PRETRAINED_VOCAB_FILES_MAP
    pretrained_init_configuration = PRETRAINED_INIT_CONFIGURATION
    max_model_input_sizes = PRETRAINED_POSITIONAL_EMBEDDINGS_SIZES

    nucleotide_abc = {"A", "T", "C", "G"}
    extended_nucleotide_abc = {"A", "T", "C", "G", "*"}
    sequence_unk_token = 'N'

    default_unk_token = "[UNK]"
    default_sep_token = "[SEP]"
    default_pad_token = "[PAD]"
    default_cls_token = "[CLS]"
    default_mask_token = "[MASK]"

    vocab_files_names = {"vocab_file": "vocab.txt"}


    def __init__(
        self,
        vocab_file: Optional[str] = None,
        kmer: int = 6,
        shift: int = 1,
        operation_space: str = "kmer",
        **kwargs,
    ):


        self.config = {}
        resolved_vocab_file = resolve_vocab_file(vocab_file, kmer)
        self.vocab = load_vocab(resolved_vocab_file)

        self.id2token = {v: k for k, v in self.vocab.items()}
        self.kmer = kmer
        self.shift = shift
        self.operation_space = operation_space

        self.config["kmer"] = kmer
        self.config["shift"] = shift
        self.config["operation_space"] = operation_space


        kwargs.setdefault("cls_token", "[CLS]")
        kwargs.setdefault("sep_token", "[SEP]")
        kwargs.setdefault("pad_token", "[PAD]")
        kwargs.setdefault("unk_token", "[UNK]")
        kwargs.setdefault("mask_token", "[MASK]")
        self.special_tokens = [kwargs["cls_token"], kwargs["sep_token"], kwargs["pad_token"], kwargs["unk_token"], kwargs["mask_token"]]
        super().__init__(**kwargs)
        if self.operation_space == 'sequence':
            token_extension = sorted(list(set(generate_kmers(LCATokenizer.extended_nucleotide_abc, self.config['kmer'])) - \
                 set(generate_kmers(LCATokenizer.nucleotide_abc, self.config['kmer'])) ))
            self.extended_vocab = deepcopy(self.vocab)
            for token in token_extension:
                self.extended_vocab[token] = 4

            self.unk_token = LCATokenizer.sequence_unk_token * self.config['shift']
            self.mask_token = '*'
            self.extended_vocab[self.mask_token] = self.vocab['[MASK]']

            full_unk = 'N' * self.config['kmer']
            self.vocab[full_unk] = 1
            self.id2token[1] = full_unk
            self.full_unk_token = full_unk

        else:
            self.extended_vocab = self.vocab
            self.unk_token = '[UNK]'

        self.unkown_tokenid = self.vocab['[UNK]']
        self.sep_token = '[SEP]'
        self.cls_token = '[CLS]'
        self.pad_token = '[PAD]'
        self.mask_token = '[MASK]'
        self.special_tokens = list(self.special_tokens_map.values())


    def get_vocab(self) -> Dict[str, int]:
        return self.vocab


    def _tokenize(self, text, **kwargs):


        offset = kwargs.get("offset", 0)


        return self.lca_kmer_tokenize_segment(text, offset)

    def _convert_token_to_id(self, token: str) -> int:


        return self.extended_vocab.get(token, self.unkown_tokenid)

    def _convert_id_to_token(self, index: int) -> str:


        return self.id2token.get(index, self.unk_token)

    def __len__(self) -> int:


        return len(self.vocab)

    def lca_kmer_tokenize_segment(self, segment: str, offset: int):

        shift = self.shift
        kmer = self.kmer


        kmers = [segment[i:i + kmer] for i in range(offset, len(segment) - kmer + 1, shift)]

        return kmers

    def tokenize(self, text: str, **kwargs) -> List[str]:


        return self._tokenize(text, **kwargs)

    def encode(self, text: str,  **kwargs) -> List[int]:


        offset = kwargs.get("offset", 0)
        kwargs["offset"] = offset
        return super().encode(text, **kwargs)

    def build_inputs_with_special_tokens(
        self, token_ids_0: List[int], token_ids_1: Optional[List[int]] = None
    ) -> List[int]:


        if token_ids_1 is None:
            return [self.cls_token_id] + token_ids_0 + [self.sep_token_id]

        input_ids = [self.cls_token_id] + token_ids_0 + [self.sep_token_id] + token_ids_1 + [self.sep_token_id]

        return input_ids

    def create_token_type_ids_from_sequences(
        self, token_ids_0: List[int], token_ids_1: Optional[List[int]] = None
    ) -> List[int]:


        if token_ids_1 is None:
            return (len(token_ids_0)+2) * [0]
        return [0] * len(token_ids_0) + [1] * len(token_ids_1)

    def batch_encode_plus(self, *args, **kwargs):


        act_outputs = super().batch_encode_plus(*args, **kwargs)
        return act_outputs


    def save_vocabulary(self, save_directory: str, filename_prefix: Optional[str] = None) -> Tuple[str]:


        if filename_prefix is None:
            filename_prefix = ""
        vocab_file_path = os.path.join(save_directory, filename_prefix + "vocab.txt")
        with open(vocab_file_path, "w") as f:
            for token in self.vocab:
                f.write(token + "\n")
        return (vocab_file_path,)


    @property
    def vocab_size(self) -> int:


        return len(self.vocab)

    def save_pretrained(self, save_directory: str, **kwargs):


        if not os.path.exists(save_directory):
            os.makedirs(save_directory)


        super().save_pretrained(save_directory, **kwargs)


        tokenizer_config_path = os.path.join(save_directory, "tokenizer_config.json")


        if os.path.exists(tokenizer_config_path):
            with open(tokenizer_config_path, "r", encoding="utf-8") as f:
                tokenizer_config = json.load(f)
        else:
            tokenizer_config = {}


        tokenizer_config["kmer"] = self.kmer
        tokenizer_config["shift"] = self.shift
        tokenizer_config["operation_space"] = self.operation_space

        with open(tokenizer_config_path, "w", encoding="utf-8") as f:
            json.dump(tokenizer_config, f, indent=2)
