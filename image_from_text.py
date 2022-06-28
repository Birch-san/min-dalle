import argparse
import os
from PIL import Image
import time
from typing import Union

from numpy import ndarray
import torch
from torch import LongTensor, FloatTensor

from min_dalle.min_dalle_torch import encode_torch, generate_image_tokens_torch
from min_dalle.min_dalle_flax import generate_image_tokens_flax
from min_dalle.generate_image import generate_image_from_text, get_img_dependencies, detokenize_torch, ImgDeps, ImageTokensBackend, GenerateImageTokensSpec


parser = argparse.ArgumentParser()
parser.add_argument('--mega', action='store_true')
parser.add_argument('--no-mega', dest='mega', action='store_false')
parser.set_defaults(mega=False)
parser.add_argument('--torch', action='store_true')
parser.add_argument('--no-torch', dest='torch', action='store_false')
parser.set_defaults(torch=False)
parser.add_argument('--text', type=str)
parser.add_argument('--seed', type=int, default=0)
parser.add_argument('--image_path', type=str, default='generated')
parser.add_argument('--image_token_count', type=int, default=256) # for debugging
parser.add_argument('--copies', type=int, default=1)


def ascii_from_image(image: Image.Image, size: int) -> str:
    rgb_pixels = image.resize((size, int(0.55 * size))).convert('L').getdata()
    chars = list('.,;/IOX')
    chars = [chars[i * len(chars) // 256] for i in rgb_pixels]
    chars = [chars[i * size: (i + 1) * size] for i in range(size // 2)]
    return '\n'.join(''.join(row) for row in chars)


def save_image(image: Image.Image, path: str):
    if os.path.isdir(path):
        path = os.path.join(path, 'generated.png')
    elif not path.endswith('.png'):
        path += '.png'
    print("saving image to", path)
    image.save(path)
    return image

if __name__ == '__main__':
    args = parser.parse_args()

    print(args)

    print("preparing dependencies...")

    start_deps = time.perf_counter()
    deps: ImgDeps = get_img_dependencies(
        text = args.text,
        is_mega = args.mega
    )

    end_deps = time.perf_counter()
    print(f"prepared dependencies in {end_deps - start_deps} secs")

    (text_tokens_arr, config, params_dalle_bart) = deps
    text_tokens: LongTensor = torch.tensor(text_tokens_arr).to(torch.long)
    encoder_state: FloatTensor = encode_torch(
        text_tokens, 
        config, 
        params_dalle_bart
    )

    def torch_backend(spec: GenerateImageTokensSpec) -> Union[ndarray, None]:
        (seed,) = spec
        image_tokens = generate_image_tokens_torch(
            text_tokens = text_tokens,
            seed = seed,
            config = config,
            params = params_dalle_bart,
            image_token_count = args.image_token_count
        )
        if args.image_token_count != config['image_length']:
            return None
        image = detokenize_torch(image_tokens)
        return image

    def flax_backend(spec: GenerateImageTokensSpec) -> ndarray:
        (seed,) = spec
        image_tokens = generate_image_tokens_flax(
            text_tokens = text_tokens, 
            seed = seed,
            config = config,
            params = params_dalle_bart,
        )
        image = detokenize_torch(torch.tensor(image_tokens))
        return image

    image_tokens_backend: ImageTokensBackend = torch_backend if args.torch else flax_backend

    start_imgs = time.perf_counter()
    print(f"generating {args.copies} images, of prompt '{args.text}'.")
    for copy in range(0, args.copies):
        start_img = time.perf_counter()
        generate_image_tokens_spec = GenerateImageTokensSpec(
            seed = args.seed + copy
        )
        image_arr: Union[ndarray, None] = image_tokens_backend(generate_image_tokens_spec)
        image = None if image_arr is None else Image.fromarray(image_arr)
        end_img = time.perf_counter()
        print(f"generated image {copy} in {end_img - start_img} secs)")
        print(f"that's {(end_img - start_img)/60} mins)")
        
        if image != None:
            save_image(image, f'./out/{args.text}.{copy}.png')
            print(ascii_from_image(image, size=128))
    end_imgs = time.perf_counter()
    print(f"prepared {args.copies} images in {end_imgs - start_imgs} secs")
    print(f"took {(end_imgs - start_imgs)/args.copies} secs per img")
    print(f"that's {((end_imgs - start_imgs)/args.copies)/60} mins per img")