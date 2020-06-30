# Serve a TorchScript model

The model MUST be a TorchScript model, to convert pyTorch models see below.

Example of a `manifest.json`:
```json
{
  "type": "torch_script",
  "saved_model_uri": "model.ts",
  "inputs": [
    {
      "name": "input",
      "shape": [
        -1,
        10
      ],
      "type": "long"
    }
  ],
  "outputs": [
    {
      "name": "output",
      "shape": [
        1,
        2
      ],
      "type": "float"
    }
  ]
}
```

# Convert pyTorch to TorchScript

Use `converter/convert.py`.  
You may have to modify the generated manifest.json to fit your needs.

```
usage: convert.py [-h] model output_folder input_examples

Convert pyTorch models into TorchScript models

positional arguments:
  model           pyTorch model (e.g. model.pt)
  output_folder   TorchScript model and manifest will be saved there
  input_example   JSON used as input for the model (e.g. [[0.5, 0.2]])

optional arguments:
  -h, --help      show this help message and exit
```
