# Build JNI binding

```bash
make -C huggingface-tokenizer-jni build
```

# Serve an HuggingFace model

Example of a `manifest.json`:
```json
{
  "type": "flow",
  "evaluator_manifests": [
    {
      "type": "huggingface_tokenizer",
      "saved_model_uri": "tokenizer.json"
    },
    {
      "type": "torch_script",
      "saved_model_uri": "transformer.ts",
      "inputs": [
        {
          "name": "ids",
          "shape": [
            1,
            -1
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
  ]
}
```

# Convert HuggingFace tokenizer and transformer for Serving

Use `converter/convert.py`.  
You may have to modify the generated manifest.json to fit your needs.

```
usage: convert.py [-h] -t TOKENIZER_CLASS -r TRANSFORMER_CLASS -i INPUT -o
                  OUTPUT

optional arguments:
  -h, --help            show this help message and exit
  -t TOKENIZER_CLASS, --tokenizer_class TOKENIZER_CLASS
                        Tokenizer class (e.g. BertWordPieceTokenizer)
  -r TRANSFORMER_CLASS, --transformer_class TRANSFORMER_CLASS
                        Transformer class (e.g. BertForSequenceClassification)
  -i INPUT, --input INPUT
                        Input model and configuration folder
  -o OUTPUT, --output OUTPUT
                        Output model and configuration folder
```
