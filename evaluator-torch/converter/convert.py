#!/usr/bin/env python3

import argparse
import json
import os

import torch
from tqdm import tqdm

TORCH_SCRIPT_FILE = 'model.ts'
MANIFEST_FILE = 'manifest.json'


def build_manifest(model_uri, input_example, output_example):
    return {
        'type': 'torch_script',
        'saved_model_uri': model_uri,
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


def convert(model_input, output_folder, input_example):
    try:
        os.mkdir(output_folder)
    except FileExistsError:
        pass

    with tqdm(bar_format='{desc}Took {elapsed}') as progress:
        progress.set_description('Loading pyTorch model {}'.format(model_input))
        model = torch.load(model_input)

        progress.set_description('Detect input and output shape and type')
        output_example = model(*input_example)
        if isinstance(output_example, torch.Tensor):
            output_example = (output_example,)

        progress.set_description('Converting pyTorch model to TorchScript model')
        traced_model = torch.jit.trace(model, input_example)

        progress.set_description('Saving TorchScript model {}'.format(TORCH_SCRIPT_FILE))
        torch.jit.save(traced_model, TORCH_SCRIPT_FILE)

        manifest_uri = '{}/{}'.format(output_folder, MANIFEST_FILE)
        progress.set_description('Saving {}'.format(manifest_uri))
        with open(manifest_uri, 'w') as f:
            manifest = build_manifest(TORCH_SCRIPT_FILE, input_example, output_example)
            tqdm.write('Model inputs:')
            for tensor in manifest['inputs']:
                tqdm.write('  name={} shape={} type={}'.format(tensor['name'], tensor['shape'], tensor['type']))
            tqdm.write('Model outputs:')
            for tensor in manifest['outputs']:
                tqdm.write('  name={} shape={} type={}'.format(tensor['name'], tensor['shape'], tensor['type']))
            json.dump(manifest, f, indent=2)

        progress.set_description('Success ! You may have to modify {} to fit your need'.format(manifest_uri))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Convert pyTorch models into TorchScript models')
    parser.add_argument('model', help='pyTorch model (e.g. model.pt)')
    parser.add_argument('output_folder', help='TorchScript model and manifest will be saved there')
    parser.add_argument('input_example', help='JSON used as input for the model (e.g. [[0.5, 0.2]])')
    args = parser.parse_args()

    convert(
        args.model,
        args.output_folder,
        [
            torch.tensor(tensor)
            for tensor in json.loads(args.input_example)
        ],
    )
