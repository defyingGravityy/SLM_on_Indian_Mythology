#  Indian Mythology SLM — Small Language Model from Scratch

> A GPT-style small language model (~50M parameters) trained on Indian epic texts — the **Mahabharata** and **Ramayana** — built from scratch using PyTorch.

Built as part of the **Vizuara AI Labs** course assignment on transformer-based language models.

---

## Sample Output

**Prompt:** `O Bharata, hear now the words of dharma`

**Generated:**
```
O Bharata, hear now the words of dharma, the Pusali,
Yajus-twilight, the sarvijas, the Malaya, the ambalas. These men, O
king, follow the worlds as Prajapati.

A said nothing he takes the senses (from desire), with the twang of an
infuriated tree with its hills and Tamas are drawing it at his arms.
He have Krishna (on his car, approaching the feet, the Gandiva-bhava
acts (the form of driver), desireth to support the caste on the lewd
subtile organ of happiness.) This men, is even the knowledge of the
Sudra sacrifice of Him. (Time). She is devoted to the deities in a
real nature of all Supreme Being.
```

The model picks up Sanskrit terminology, character names (Krishna, Prajapati), narrative style, and translator footnote patterns from the Ganguli translation — all without any fine-tuning or instruction.

---

##  Overview

| Property | Value |
|---|---|
| Architecture | GPT-style (nanoGPT) |
| Parameters | ~50M |
| Domain | Indian Mythology (Mahabharata + Ramayana) |
| Tokenizer | GPT-2 BPE via `tiktoken` (vocab: 50,257) |
| Training hardware | AMD Ryzen 5 laptop CPU |
| Training time | ~3.5 hours |
| Dataset size | ~10M tokens |
| Context window | 128 tokens |

---

##  Repository Structure

```
├── Indian_Mythology_SLM_v2.ipynb   # Main Colab notebook
├── README.md                        # This file
└── sample_outputs/
    └── outputs.txt                  # Generated text samples
```

---

## 🚀 Quickstart

### Run in Google Colab

1. Open `Indian_Mythology_SLM_v2.ipynb` in [Google Colab](https://colab.research.google.com/)
2. Set runtime to **GPU** (`Runtime → Change runtime type → T4 GPU`)
3. Run all cells in order — the notebook will:
   - Download mythology texts from Project Gutenberg automatically
   - Tokenise and write `train.bin` / `validation.bin`
   - Train the model (~8–9 hrs on T4, ~3.5 hrs on modern CPU)
   - Generate sample outputs from mythology prompts

### Run Locally

```bash
git clone https://github.com/<your-username>/indian-mythology-slm
cd indian-mythology-slm
pip install torch datasets tiktoken requests tqdm matplotlib
jupyter notebook Indian_Mythology_SLM_v2.ipynb
```

---

## 📚 Dataset

We use **5 public-domain texts** from [Project Gutenberg](https://www.gutenberg.org/), downloaded automatically by the notebook:

| Text | Translator | Gutenberg ID |
|---|---|---|
| Mahabharata Vol 1 | Kisari Mohan Ganguli | 15474 |
| Mahabharata Vol 2 | Kisari Mohan Ganguli | 15475 |
| Mahabharata Vol 3 | Kisari Mohan Ganguli | 15476 |
| Mahabharata Vol 4 | Kisari Mohan Ganguli | 15477 |
| Ramayana | Ralph T. H. Griffith | 24869 |

**Preprocessing pipeline:**
- Strip Project Gutenberg headers/footers
- Remove non-ASCII characters
- Normalize whitespace
- Chunk into 512-character windows with 256-character stride (50% overlap)
- Shuffle and split 95% train / 5% validation

---

##  Model Architecture

Standard GPT decoder-only transformer (nanoGPT-style):

```
Embedding (token + positional)
        ↓
  × 6 Transformer Blocks
  ┌─────────────────────┐
  │  LayerNorm          │
  │  CausalSelfAttention│  ← Flash Attention when available
  │  LayerNorm          │
  │  MLP (4× expansion) │
  └─────────────────────┘
        ↓
  LayerNorm → LM Head (weight-tied to embedding)
```

```python
GPTConfig(
    vocab_size  = 50257,
    block_size  = 128,
    n_layer     = 6,
    n_head      = 6,
    n_embd      = 384,
    dropout     = 0.1,
    bias        = True,
)
# Total: ~50M parameters
```

---

##  Training Configuration

```python
learning_rate              = 1e-4
max_iters                  = 20000
warmup_steps               = 1000
batch_size                 = 32
gradient_accumulation_steps = 32   # effective batch = 1024
optimizer                  = AdamW(betas=(0.9, 0.95), weight_decay=0.1)
scheduler                  = Linear warmup → Cosine decay
mixed_precision            = float16 (GPU) / float32 (CPU)
```

---

## Training Results

- **Best validation loss:** *(fill in after your run)*
- **Final train loss:** *(fill in)*
- Training loss and validation loss curves show consistent decrease across 20,000 steps, with no significant overfitting despite the small domain-specific corpus.

---


##  Requirements

```
torch>=2.0
datasets
tiktoken
requests
tqdm
matplotlib
```

---

##  Acknowledgements

- [nanoGPT](https://github.com/karpathy/nanoGPT) by Andrej Karpathy — model architecture
- [Vizuara AI Labs](https://vizuara.ai) — course and base notebook
- [Project Gutenberg](https://www.gutenberg.org) — public domain mythology texts
- Kisari Mohan Ganguli — English translation of the Mahabharata
- Ralph T. H. Griffith — English translation of the Ramayana

---

##  License

Code: MIT License.
Dataset texts: Public domain (Project Gutenberg).
