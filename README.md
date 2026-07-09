<<<<<<< HEAD
# Mini-GPT with Byte-Level and BPE Tokenization

A small GPT-style language model implemented in PyTorch and trained on Tiny Shakespeare.

The project covers the complete language-modeling pipeline:

```text
raw text
→ tokenization
→ next-token dataset
→ decoder-only Transformer
→ training and validation
→ checkpointing
→ autoregressive text generation
```

Two tokenization modes are supported:

- **Byte-level tokenization**, using UTF-8 bytes.
- **A custom BPE tokenizer**, trained from scratch and stored as JSON.

The main purpose of the project is educational: understanding how tokenization, causal masking, Transformer training, regularization, evaluation, and autoregressive generation interact in a small language model.

---

## Features

### Tokenization

- UTF-8 byte-level encoding and decoding.
- Custom character-based BPE implementation.
- Ranked BPE merge rules.
- BPE vocabulary and rules saved to JSON.
- Dynamic vocabulary size according to the selected tokenizer.
- Token ID `0` reserved for possible future padding support.
- Encode/decode round-trip tests for both tokenization modes.

### Model

- Decoder-only GPT-style Transformer.
- Learned token embeddings.
- Learned positional embeddings.
- Causal self-attention mask.
- Multi-head self-attention.
- Feed-forward layers.
- Configurable vocabulary size and model dimensions.
- Dropout regularization.
- Xavier initialization for Transformer weights.

### Training

- Next-token prediction with cross-entropy loss.
- AdamW optimizer.
- Linear learning-rate warmup.
- Cosine learning-rate decay.
- Gradient clipping.
- Training by epochs, optimizer steps, or both.
- Periodic train and validation evaluation.
- Best and latest checkpoints.
- CSV metric logging.
- Automatic loss and accuracy plots.
- Reproducible random seed configuration.

### Evaluation and generation

- Validation loss.
- Token-level validation accuracy.
- Loss per character for comparing different tokenizers.
- Average characters represented per token.
- Autoregressive text generation.
- Greedy decoding.
- Temperature scaling.
- Top-k sampling.
- Top-p nucleus sampling.
- Predefined generation presets.

---

## Project Structure

```text
.
├── src/
│   ├── data.py
│   ├── models.py
│   ├── tokenizer.py
│   ├── train.py
│   ├── utils.py
│   └── viz.py
├── tests/
│   ├── test_byte_level_overfit.py
│   ├── test_data_shape.py
│   ├── test_mask.py
│   ├── test_roundtrip_BPE_encode_decode.py
|   ├── test_roundtrip_BPE_encode_decode2.py
│   ├── test_roundtrip_byte_level_encode_decode.py
│   ├── test_roundtrip_json_tokenizer.py
│   └── test_shift_data.py
├── tokenizations/
├── runs/
├── main.py
├── train_tokenizer.py
├── tiny_shakespare.txt (data file)
├── requirements.txt
└── README.md
```

All commands in this README assume that they are executed from the project root, however it should work anywhere if it is used with the correspondent correct absolute or relative path.

---

## Model Architecture

The model is a small decoder-only Transformer built with PyTorch's `TransformerEncoder` layers and a causal attention mask.

Although `TransformerEncoder` is used internally, the causal mask prevents every position from attending to future tokens, so the stack behaves as a decoder-only autoregressive model.

Current default model configuration:

| Parameter | Default |
|---|---:|
| Vocabulary size | 257 if tokenenization-name is not provided else is used the tokenization one |
| Embedding dimension | 256 |
| Transformer layers | 6 |
| Attention heads | 8 |
| Feed-forward dimension | 1024 |
| Maximum positional context | 256 tokens |
| Dropout | 0.1 |

The forward pass is:

```text
token IDs
→ token embeddings
→ positional embeddings
→ causal Transformer stack
→ linear vocabulary projection
→ next-token logits
```

The output shape is:

```text
[batch_size, sequence_length, vocabulary_size]
```

At every position, the model predicts the token that follows the current input token.

---

## Tokenization

## Byte-level mode

Byte-level mode encodes the UTF-8 representation of the text.

Each byte is shifted by one:

```text
byte value 0   → token ID 1
byte value 255 → token ID 256
```

Token ID `0` is reserved for possible padding support, although padding is not currently used.

The byte-level vocabulary therefore contains:

```text
257 IDs = 1 reserved ID + 256 byte IDs
```

Byte-level mode is used when `--tokenization-name` is omitted.

---

## Custom BPE mode

The project includes a custom BPE tokenizer implemented from scratch.

Its training process is:

1. Split the corpus into alphabetic/apostrophe pieces and non-alphabetic pieces.
2. Represent each piece initially as characters.
3. Count adjacent token-pair frequencies across the corpus.
4. Merge the most frequent pair.
5. Repeat until the requested number of merges has been learned or there aren't any possible merges left.
6. Assign every merge rule a rank according to its creation order.
7. Save the vocabulary and merge rules to JSON.

During encoding, the available merge rule with the lowest rank is applied first.

The generated JSON stores:

```text
tokenization name
creation date
vocabulary size
token-to-ID mapping
ID-to-token mapping
ranked merge rules
```

> **Implementation note:** this is a character-based BPE implementation whose initial vocabulary contains code points from `0` to `255`. It is not a full byte-level BPE implementation operating directly over UTF-8 byte sequences. Characters outside the initial range may require additional handling.

---

## Training a BPE Tokenizer

Create a tokenizer with 100 merges:

```bash
python train_tokenizer.py \
  --num-merges 100 \
  --tokenization-name 100_merges_tokenization.json \
  --text-path tiny_shakespare.txt
```

Available arguments:

| Argument | Description | Default |
|---|---|---|
| `--num-merges` | Maximum number of merge rules to learn | `100` |
| `--tokenization-name` | Output JSON filename | Timestamped name |
| `--text-path` | Training text path | `tiny_shakespare.txt` |
| `--save-dir` | Custom output directory | `tokenizations/` |

When no custom directory is supplied, the tokenizer is saved under `tokenizations/` folder, if it not exists it will be created in the root of the project.

---

## Dataset

The corpus is encoded using either byte-level or BPE tokenization and then divided into training and validation portions.

Current data configuration:

| Parameter | Value |
|---|---:|
| Training split | 90% |
| Validation split | 10% |
| Training window | 128 tokens |
| Stride | 8 tokens |

Training examples are created with overlapping sliding windows:

```text
X = tokens[t     : t + window]
Y = tokens[t + 1 : t + window + 1]
```

The starting index advances by `stride`:

```text
t = 0, stride, 2 × stride, 3 × stride, ...
```

Therefore, `Y` is the same sequence as `X`, shifted one token to the right.

---

## Installation

Create and activate a virtual environment:

```bash
python -m venv venv
source venv/bin/activate
```

Install the dependencies:

```bash
pip install -r requirements.txt
```

For GPU training, install the PyTorch build that matches the available CUDA environment.

---

## Training

At least one training limit must be supplied:

```text
--max-steps
--epochs
```

If both are supplied, training stops when the first limit is reached.

### Byte-level training

```bash
python main.py \
  --max-steps 30000 \
  --batch-size 192 \
  --lr 14e-4 \
  --weight-decay 0.1
```

### BPE training

```bash
python main.py \
  --max-steps 30000 \
  --batch-size 192 \
  --lr 14e-4 \
  --weight-decay 0.1 \
  --tokenization-name 100_merges_tokenization.json
```

A tokenizer filename is resolved from the default `tokenizations/` directory. An absolute path can also be supplied.

Main training arguments:

| Argument | Description | Default |
|---|---|---:|
| `--epochs` | Maximum number of epochs | `None` |
| `--max-steps` | Maximum number of optimizer steps | `None` |
| `--batch-size` | Batch size | `32` |
| `--lr` | AdamW learning rate | `3e-4` |
| `--weight-decay` | AdamW weight decay | `0.1` |
| `--device` | Training device | CUDA if available, otherwise CPU |
| `--ckpt-path` | Checkpoint to load before training | `None` |
| `--tokenization-name` | BPE JSON filename or path | Byte-level mode |

The model is evaluated every 500 optimizer steps. (Adjustable by changing the value of `N_STEPS` in the fit method in `train.py.`)

The learning-rate schedule consists of:

```text
first 5% of planned steps → linear warmup
remaining steps           → cosine annealing (scheduler)
```

Gradients are clipped to a maximum norm of `0.5`.

---

## Run Artifacts

Every training run creates a timestamped directory:

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

### `metrics.csv`

Stores periodic training and validation information:

```text
step
split
loss
accuracy
learning rate
duration
```

### `features.md`

Stores model and training configuration such as:

```text
model dimensions
dropout
initial learning rate
batch size
weight decay
tokenization filename
run date
```

### Checkpoints

```text
best.pt → lowest validation-loss checkpoint
last.pt → latest checkpoint
```

A checkpoint stores:

- model weights;
- model class name;
- model configuration;
- optimizer state;
- training step;
- step/epoch mode metadata;
- tokenization filename.

The model can therefore be reconstructed automatically without manually reproducing its architecture.

The current implementation does **not** restore the warmup or scheduler state, so loading a checkpoint should not be considered an exact continuation of the original training schedule.

---

## Evaluation

### Evaluate a byte-level checkpoint

```bash
python main.py \
  --eval-only \
  --eval-metrics \
  --ckpt-path "runs/<RUN_NAME>/best.pt"
```

### Evaluate a BPE checkpoint

```bash
python main.py \
  --eval-only \
  --eval-metrics \
  --ckpt-path "runs/<RUN_NAME>/best.pt" \
  --tokenization-name 100_merges_tokenization.json
```

The tokenizer supplied during evaluation must match the tokenizer used to train the checkpoint.

The evaluation output contains:

```text
validation loss
validation token accuracy
loss per character
average characters per token
```

---

## Comparing Different Tokenizers

Raw token-level loss is not directly comparable between byte-level and BPE models because their tokens represent different amounts of text.

For example:

```text
byte-level token → one byte
BPE token        → may represent several characters
```

The project therefore computes a normalized loss per character:

```text
loss_per_character =
    validation_loss_per_token × number_of_target_tokens
    ---------------------------------------------------
                  number_of_decoded_characters
```

It also reports:

```text
average_characters_per_token =
    number_of_decoded_characters
    ----------------------------
      number_of_target_tokens
```

Token accuracy should also be interpreted cautiously across tokenizers: predicting one long BPE token is not equivalent to predicting one byte token.

---

## Current Experiment Results

The following results were obtained with the same core model architecture on Tiny Shakespeare.

| Tokenization | Merges | Dropout | Steps | Eval loss/token | Loss/character | Characters/token |
|---|---:|---:|---:|---:|---:|---:|
| Byte-level | — | 0.1 | 30000 | 1.4703 | 1.4703 | 1.0000 |
| Byte-level | — | 0.3 | 12000 | 1.4590 | 1.4590 | 1.0000 |
| BPE | 100 | 0.1 | 30000 | 2.1988 | 1.4866 | 1.4790 |
| BPE | 100 | 0.2 | 30000 | 2.1782 | 1.4727 | 1.4790 |
| BPE | 100 | 0.3 | 12000 | 2.1288 | 1.4394 | 1.4790 |
| BPE | 500 | 0.1 | 30000 | 2.8434  | 1.5313 | 1.8569 |

These runs are useful for exploration, but they are not a perfectly controlled benchmark: some configurations use different dropout values and training budgets.

Main observations:

- BPE reduces sequence length and increases the amount of text represented by each token.
- Greater compression does not automatically produce better validation performance.
- BPE with 500 merges compressed more than BPE with 100 merges, but generalized worse.
- BPE with 100 merges became more competitive as dropout increased.
- The best result obtained so far is BPE with 100 merges and `0.3` dropout.
- Byte-level tokenization remains a strong baseline for a small corpus because its vocabulary is compact and its tokens receive many training updates.
- Regularization and tokenizer granularity interact strongly in this setup.
- For a better BPE performance, it seems that a larger dataset is needed.

The main model-selection criterion is validation loss. Loss per character is used when comparing different tokenization schemes and it only reported when the model is run with both the `--eval-only` and `--eval-metrics` CLI options.

---

## Text Generation

### Byte-level generation

```bash
python main.py \
  --eval-only \
  --ckpt-path "runs/<RUN_NAME>/best.pt" \
  --prompt "ROMEO:" \
  --preset short_stable \
  --max-new-tokens 300
```

### BPE generation

```bash
python main.py \
  --eval-only \
  --ckpt-path "runs/<RUN_NAME>/best.pt" \
  --tokenization-name 100_merges_tokenization.json \
  --prompt "ROMEO:" \
  --preset short_stable \
  --max-new-tokens 300
```

Generation arguments:

| Argument | Description | Default |
|---|---|---|
| `--prompt` | Initial text prompt | Empty |
| `--max-new-tokens` | Number of generated tokens | `128` |
| `--preset` | Sampling preset | `default` |
| `--temperature` | Manual temperature override | Preset value |
| `--top-k` | Keep only the `k` most likely tokens | Disabled |
| `--top-p` | Nucleus-sampling probability threshold | Preset value |

Available presets:

| Preset | Behavior |Temperature|Top p|
|---|---|---|---|
| `debug_greedy` | Deterministic argmax decoding | `1` | None |
| `short_stable` | More conservative short-form generation | `0.9` | `0.9` |
| `default` | General-purpose sampling | `0.92` | `0.95` |
| `creative` | Higher-temperature, more varied generation | `1.1` | `0.95`|

Manual CLI values override the corresponding preset values.

During generation, only the most recent 128 token IDs of the `--prompt` entry are passed back into the model. The complete generated token sequence is decoded once generation finishes.

---

## Running the Tests

Run the complete test suite:

```bash
python -m pytest
```

The tests cover:

- byte-level encode/decode round trips;
- BPE encode/decode round trips;
- tokenizer JSON save/load consistency;
- dataset shape;
- next-token target shifting;
- causal masking;
- basic overfitting on a small training example.

The tests are intended to detect structural pipeline errors rather than measure final language quality.

---

## Limitations

This is an educational implementation, not a production language model.

Current limitations include:

- Tiny Shakespeare is a small and highly specialized corpus.
- Generated text reproduces style and formatting more reliably than long-range semantic coherence.
- The model is a continuation model, not an instruction-following assistant.
- The BPE tokenizer may require additional handling for characters outside its initial vocabulary.
- More BPE merges can produce sparse tokens that receive relatively few training updates.
- Exact training resumption is not implemented because scheduler and warmup state are not restored.
- Padding ID `0` is reserved but padding is not currently implemented.
- No KV-cache is used during autoregressive generation.
- Generation recomputes the complete active context after every new token.

---

## Possible Future Work

- Implement true byte-level BPE over UTF-8 byte sequences.
- Implement functional padding for dynamic batch_size training.
- Add explicit unknown-token or fallback handling to the character-based BPE tokenizer.
- Implement a KV-cache for faster generation.
- Implement the Transformer blocks directly instead of delegating them to `torch.nn.TransformerEncoder`.

---

## Educational Scope

This project intentionally prioritizes transparency and experimentation over production efficiency.

It was built to study:

- how next-token datasets are created;
- why causal masking is necessary;
- how tokenization affects sequence length and vocabulary size;
- why raw token loss cannot always be compared across tokenizers;
- how dropout changes generalization;
- how temperature, top-k, and top-p affect generation.

A central result of the project is that a more complex tokenizer does not automatically produce a better model. On a small corpus, compression, vocabulary sparsity, regularization, and model capacity must be considered together.

---

## Disclaimer

The generated text may imitate the structure and style of the training corpus, but the model does not possess semantic understanding and may produce incoherent, incorrect, or invented content.

---

## License

MIT License.
=======
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
## Notes

- This project assumes that all commands are executed from the project root ( .../Mini-GPT$ )

---
## Disclaimer

This project is a learning implementation.

The generated text can imitate the style and structure of the training data, but it does not have deep semantic understanding and can produce incoherent or invented words.

---

## License

MIT License.

>>>>>>> main
