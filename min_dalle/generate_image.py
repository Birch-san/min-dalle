import os
import json
import numpy
from numpy import ndarray
from torch import FloatTensor
from PIL import Image
from typing import Tuple, List, NamedTuple, Dict, Callable, Union
from typing_extensions import TypeAlias
import torch

from min_dalle.load_params import load_dalle_bart_flax_params
from min_dalle.text_tokenizer import TextTokenizer
from min_dalle.min_dalle_flax import generate_image_tokens_flax
from min_dalle.min_dalle_torch import (
    generate_image_tokens_torch,
    detokenize_torch
)

def load_dalle_bart_metadata(path: str) -> Tuple[dict, dict, List[str]]:
    print("parsing metadata from {}".format(path))
    for f in ['config.json', 'flax_model.msgpack', 'vocab.json', 'merges.txt']:
        assert(os.path.exists(os.path.join(path, f)))
    with open(path + '/config.json', 'r') as f: 
        config = json.load(f)
    with open(path + '/vocab.json') as f:
        vocab = json.load(f)
    with open(path + '/merges.txt') as f:
        merges = f.read().split("\n")[1:-1]
    return config, vocab, merges

def tokenize_text(
    text: str, 
    config: dict,
    vocab: dict,
    merges: List[str]
) -> numpy.ndarray:
    print("tokenizing text")
    tokens = TextTokenizer(vocab, merges)(text)
    print("text tokens", tokens)
    text_tokens = numpy.ones((2, config['max_text_length']), dtype=numpy.int32)
    text_tokens[0, :len(tokens)] = tokens
    text_tokens[1, :2] = [tokens[0], tokens[-1]]
    return text_tokens

class ImgDeps(NamedTuple):
    text_tokens_arr: numpy.ndarray
    config: dict
    params_dalle_bart: Dict[str, ndarray]

class GenerateImageTokensSpec(NamedTuple):
    seed: int

ImageTokensBackend: TypeAlias = Callable[[GenerateImageTokensSpec], Union[ndarray, None]]

def get_img_dependencies(
    text: str,
    is_mega: bool = False,
) -> ImgDeps:
    model_name = 'mega' if is_mega else 'mini'
    model_path = './pretrained/dalle_bart_{}'.format(model_name)
    config, vocab, merges = load_dalle_bart_metadata(model_path)
    text_tokens_arr: ndarray = tokenize_text(text, config, vocab, merges)
    params_dalle_bart: Dict[str, ndarray] = load_dalle_bart_flax_params(model_path)
    return ImgDeps(text_tokens_arr=text_tokens_arr, config=config, params_dalle_bart=params_dalle_bart)

def generate_image_from_text(
    image_tokens_backend: ImageTokensBackend,
    seed: int = 0,
) -> Image.Image:
    generate_image_tokens_spec = GenerateImageTokensSpec(
        seed = seed
    )

    image_arr: Union[ndarray, None] = image_tokens_backend(generate_image_tokens_spec)
    return None if image_arr is None else Image.fromarray(image_arr)