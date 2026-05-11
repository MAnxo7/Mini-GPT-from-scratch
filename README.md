# Mini-GPT Byte-Level

A small GPT-style language model implemented from scratch in PyTorch.

This project implements a decoder-only Transformer language model trained with next-token prediction on TinyShakespeare. The model uses byte-level tokenization (`VOCAB = 256`), causal self-attention, training by steps, checkpointing, experiment logging, and autoregressive text generation with temperature, top-k and top-p sampling.

The goal of this project is educational: to understand how a small GPT-like model works internally, from dataset creation and causal masking to training dynamics and text generation.

---

## Features

- Byte-level tokenization using UTF-8 bytes
- Decoder-only Transformer architecture
- Causal self-attention mask
- Token and positional embeddings
- Multi-layer Transformer stack
- Next-token prediction with cross-entropy loss
- Training by `max_steps`
- Warmup + cosine learning-rate scheduling
- AdamW optimizer
- Gradient clipping
- Checkpointing:
  - `best.pt`
  - `last.pt`
- Metrics logging:
  - `metrics.csv`
  - `features.md`
  - training/evaluation loss plots
  - training/evaluation accuracy plots
- Text generation with:
  - greedy decoding
  - temperature
  - top-k sampling
  - top-p sampling
  - generation presets

---

## Project Structure

```text
.
├── runs/
├── src/
│   ├── data.py
│   ├── models.py
│   ├── train.py
│   ├── utils.py
│   └── viz.py
├── tests/
│   ├── test_data_shape.py
│   ├── test_mask.py
│   ├── test_overfit.py
│   ├── test_roundtrip_encode_decode.py
│   └── test_shift_data.py
├── main.py
├── tiny_shakespare.txt (datafile)
└── README.md
```
---

## Model Overview

The model is a small decoder-only GPT-style Transformer.

Default architecture:

```text
VOCAB      = 256
d_model    = 256
context_L  = 256
n_heads    = 8
n_layers   = 6
d_ff       = 4 * d_model
dropout    = 0.1
```

The model pipeline is:

```text
input byte token IDs
→ token embeddings
→ positional embeddings
→ causal Transformer stack
→ linear projection to vocabulary logits
```

The model outputs logits with shape:

```text
[batch_size, sequence_length, vocab_size]
```

During training, the logits are compared against next-token targets using cross-entropy loss.

During generation, the logits from the last position are used to sample the next token autoregressively.

---

## Dataset

The dataset is loaded from a given text file (default=./tiny_shakespare.txt) and encoded into byte-level token IDs.

Training examples are created with sliding windows:

```text
X = tokens[t : t + window]
Y = tokens[t + 1 : t + window + 1]
```

The starting index t advances by stride after each window:

t = 0, stride, 2 * stride, 3 * stride, ..

Default dataset settings:

```text
window = 128
stride = 8
eval_split = 10%
```

Every input byte sequence `X`, the target `Y` is the same "window" length sequence shifted one position to the right (= +1). 

---

## Installation

Create and activate a virtual environment (Recommended):

```bash
python -m venv venv
source venv/bin/activate
```

Install the required dependencies:

```bash
# (In the project's source)
pip install -r requirements.txt
```

If you want CUDA support, install the PyTorch build that matches your GPU and CUDA version from the official PyTorch installation instructions.

---

## Training

Example training command:

```bash
python main.py \
  --max-steps=20000 \
  --batch-size=98 \
  --lr=1e-3 \
  --weight-decay=0.1
```

Useful training arguments:

```text
--epochs          Maximum number of epochs (default = 200)
--max-steps       Maximum number of optimizer steps (default = None)
--batch-size      Batch size (default = 32)
--lr              Learning rate (default = 3e-4)
--weight-decay    AdamW weight decay (default = 0.1)
--device          Device to use: cuda or cpu (default = utils.get_device())
```

Each training run creates a timestamped folder under `runs/`:

```text
runs/YYYY-MM-DD_HH-MM-SS/
├── metrics.csv
├── features.md
├── best.pt
├── last.pt
└── figures/
    ├── loss.jpg
    └── acc.jpg
```

---

## Checkpoints

The training loop saves two checkpoints:

```text
best.pt  → checkpoint with the best validation loss
last.pt  → latest checkpoint
```

Checkpoints store:

* model weights
* optimizer state
* model type (class name)
* model configuration
* current training step
* extra training metadata (=None in the current implementation)

This allows the model to be reconstructed from the checkpoint without manually changing the architecture parameters in `main.py`.

---

## Evaluation

To evaluate a checkpoint on the validation split:

```bash
python main.py \
  --eval-only \
  --eval-metrics \
  --ckpt-path="runs/<RUN_NAME>/best.pt"
```

This prints validation loss and validation accuracy.

---

## Text Generation

Generate text from a checkpoint:

```bash
python main.py \
  --eval-only \
  --ckpt-path="runs/<RUN_NAME>/best.pt" \
  --prompt="ALEX:" \
  --preset="short_stable" \
  --max-new-tokens=300
```

Generation arguments:

```text
--prompt           Initial prompt
--max-new-tokens   Number of new tokens to generate
--preset           Generation preset
--temperature      Optional temperature override
--top-p            Optional top-p override
--top-k            Optional top-k override
```

Available presets:

| Preset         | Description                              |
| -------------- | ---------------------------------------- |
| `debug_greedy` | Deterministic greedy decoding            |
| `short_stable` | More stable generation for short prompts |
| `default`      | General-purpose generation               |
| `creative`     | More diverse generation                  |

You can also override preset values manually:

```bash
python main.py \
  --eval-only \
  --ckpt-path="runs/<RUN_NAME>/best.pt" \
  --prompt="ALEX:" \
  --temperature=0.9 \
  --top-p=0.9 \
  --max-new-tokens=300
```

---

## Example Output

**Generated with the current Mini-GPT checkpoint trained using:**
- 20000 steps
- lr = 1e-3
- weight_decay = 0.1


Prompt:

```text
ALEX:
```

Generated sample (PRESET: short_stable):

ALEX:
Ay, the father hath caused thee for me.

MENENIUS:
Good morrow, good my lord.

AUFIDIUS:
My father, he shall be the hand: I will revenge my
heart to be too famous to be so barren at once.

DUKE VINCENTIO:
I think you, sir, we are the thing to say 'she's heart.'

LADY GREY:
Masters, gentle for his h

Another prompt:

```text
The night is calm, yet every whisper carries weight; I write to you in haste, for rumor grows, and men grow bold in shadow.
```

Generated sample (PRESET: default):

```text
The night is calm, yet every whisper carries weight; I write to you in haste, for rumor grows, and men grow bold in shadow.

CORIOLANUS:
That our airy for their heads have committed him.

CORIOLANUS:
O that happy state, is most dear better hand,
It is not meant to him.

Third Servingman:
Not you so?

MENENIUS:
I know him, my lord, I have seen a woman
To speak by my brotherhood.

GLOUCESTER:
As I do dishonour, and he hat
```

The generated text is not semantically perfect, but it captures some structure of the training corpus: theatrical dialogue, character names, line breaks, punctuation, and Shakespeare-like rhythm.

---

## Running Tests

Run the test suite with:

```bash
python -m pytest
```

The tests cover core infrastructure such as:

* encode/decode roundtrip
* dataset shape
* input/target shifting
* causal masking
* basic overfitting behavior on a small batch

The goal of the tests is not to prove that the model is good, but to verify that the training and generation pipeline is not structurally broken.

---

## Experiment Notes

Several experiments were run to understand the behavior of the byte-level model.

Main observations:

* Learning rate had a strong impact on training quality.
* Longer training improved validation loss and text quality.
* Short runs were not enough to judge architectural changes reliably.
* Weight decay was already in a reasonable range. Small variations around the baseline had little noticeable impact, while more aggressive changes tended to hurt performance.
* Increasing model size did not clearly improve validation loss under the tested setup.
* Byte-level tokenization works, but it is inefficient compared to subword tokenization.

The main checkpoint selection criterion was validation loss.

Training loss was used as a debugging signal, while validation loss was used as the primary metric for model selection.

---

## Limitations

This is an educational project, not a production language model.

Known limitations:

* Byte-level tokenization is simple but inefficient.
* The model needs many byte tokens to represent words.
* Generated text often captures style and formatting before deeper semantic coherence.
* The dataset is small and strongly biased toward Shakespeare-style dialogue.
* The model is a text continuation model, not an instruction-following assistant.
* No BPE/subword tokenizer is used yet.
* No KV-cache is implemented for fast inference.

---

## Roadmap

Possible future improvements:

* Add a BPE tokenizer branch
* Make `vocab_size` configurable for non-byte tokenizers
* Add KV-cache for faster generation
* Improve experiment tracking with a run summary file
* Add cleaner/better configuration management (with data and training attributes management)

---

## Disclaimer

This project is a learning implementation.

The generated text can imitate the style and structure of the training data, but it does not have deep semantic understanding and can produce incoherent or invented words.

---

## License

MIT License.

