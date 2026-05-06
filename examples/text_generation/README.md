# Text Generation With Binary Diffusion

![generated_text_generation.gif](https://drive.google.com/thumbnail?id=1YF_nv4SkIvdJbhlrJuCd28iybTJFS2Tq&sz=w2560)

This example shows how Densiformis can generate text without adding a text-specific generation pipeline. The core idea is to represent text as a binary tensor, then use the same diffusion API that Densiformis already uses for other data types.


Files:

- `data.txt`: A set of sentence natural-language corpus, one sentence per line.
- `train.py`: Loads the corpus, trains a small BPE tokenizer, encodes tokens as Gray-code bits, and trains a 1D CNN ResNet denoiser.
- `generate.py`: Loads a checkpoint, starts from random binary noise, denoises it into text, and writes a GIF visualization.

## Run

Train:

```bash
python examples/text_generation/train.py
```

Generate:

```bash
python examples/text_generation/generate.py
```

Outputs:

- `generated_text_samples.txt`: Decoded generated text samples.
- `generated_text_generation.gif`: A visualization of the denoising process.
- `text_generation_checkpoints/`: Model weights, optimizer state, and `text_config.json`.