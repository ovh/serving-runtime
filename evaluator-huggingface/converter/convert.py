#!/usr/bin/env python3

import argparse
import os
import json
from tqdm import tqdm

import tokenizers
import torch
import transformers

TOKENIZER_FILE = 'tokenizer.json'
TRANSFORMER_FILE = 'transformer.ts'
MANIFEST_FILE = 'manifest.json'


def build_manifest(input_example, output_example):
    return {
        'type': 'flow',
        'evaluator_manifests': [
            {
                'type': 'huggingface_tokenizer',
                'saved_model_uri': TOKENIZER_FILE
            },
            {
                'type': 'torch_script',
                'saved_model_uri': TRANSFORMER_FILE,
                'inputs': [
                    {
                        'name': 'input_{}'.format(index),
                        'shape': list(tensor.shape),
                        'type': convert_dtype(tensor.dtype),
                    }
                    for index, tensor in enumerate(input_example)
                ],
                'outputs': [
                    {
                        'name': 'output_{}'.format(index),
                        'shape': list(tensor.shape),
                        'type': convert_dtype(tensor.dtype),
                    }
                    for index, tensor in enumerate(output_example)
                ]
            }
        ]
    }


def convert_dtype(dtype):
    if dtype == torch.int32:
        return 'int'
    if dtype == torch.int64:
        return 'long'
    if dtype == torch.float32:
        return 'float'
    if dtype == torch.float64:
        return 'double'
    raise Exception('Cannot convert dtype {}'.format(dtype))


def convert(tokenizer_class, transformer_class, in_path, out_path):
    try:
        os.mkdir(out_path)
    except FileExistsError:
        pass

    with tqdm(bar_format='{desc}Took {elapsed}') as progress:
        progress.set_description('Loading tokenizer {} ...'.format(tokenizer_class))
        tokenizer = getattr(tokenizers, tokenizer_class)(vocab_file=in_path + '/vocab.txt')

        progress.set_description('Saving tokenizer ...')
        tokenizer_uri = out_path + '/' + TOKENIZER_FILE
        tokenizer.save(tokenizer_uri, pretty=True)

        progress.set_description('Encoding input example ...')
        encoding = tokenizer.encode('Example input used to trace model')
        input_example = (torch.tensor([encoding.ids]),)

        progress.set_description('Loading transformer {} ...'.format(transformer_class))
        transformer = getattr(transformers, transformer_class).from_pretrained(in_path, torchscript=True)
        transformer.eval()

        progress.set_description('Detect input and output shape and type')
        output_example = transformer(*input_example)
        if isinstance(output_example, torch.Tensor):
            output_example = (output_example,)

        progress.set_description('Converting transformer from pyTorch to TorchScript ...')
        model_traced = torch.jit.trace(transformer, input_example)

        progress.set_description('Saving TorchScript transformer ...')
        transformer_uri = out_path + '/' + TRANSFORMER_FILE
        model_traced.save(transformer_uri)

        progress.set_description('Saving manifest.json ...')
        manifest = build_manifest(input_example, output_example)
        manifest_uri = out_path + '/' + MANIFEST_FILE
        with open(manifest_uri, 'w') as f:
            json.dump(manifest, f, indent=2)

        progress.set_description('Success! You may have to modify {} to fit your need'.format(manifest_uri))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-t', '--tokenizer_class', required=True, help='Tokenizer class (e.g. BertWordPieceTokenizer)'
    )
    parser.add_argument(
        '-r', '--transformer_class', required=True, help='Transformer class (e.g. BertForSequenceClassification)'
    )
    parser.add_argument(
        '-i', '--input', required=True, help='Input model and configuration folder'
    )
    parser.add_argument(
        '-o', '--output', required=True, help='Output model and configuration folder'
    )
    args = parser.parse_args()

    convert(
        args.tokenizer_class,
        args.transformer_class,
        args.input,
        args.output,
    )
