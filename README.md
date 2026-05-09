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
├── tiny_shakespare.txt
└── README.md