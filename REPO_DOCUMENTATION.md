# autoresearch — Repository Documentation

This document explains the goals, architecture, and runtime logic of the `autoresearch` project. It is written for two audiences:

- agents (what to change and what is fixed)
- developers (how the code is organized and why things work the way they do)

**Keep this file updated when adding features, changing the experiment loop, or modifying evaluation.**

**Overview**
- **Purpose:** Run short (5-minute) autonomous experiments where an agent edits one file (`train.py`) to optimize validation bits-per-byte (`val_bpb`). The agent iterates: change code, run a fixed-time training job, evaluate, keep or discard changes.
- **Design principle:** Single editable file for experiments, fixed evaluation harness and data prep to ensure comparability across experiments.

**Key Files**
- **Repository root:** [README.md](README.md) — project intent and quick start.
- [prepare.py](prepare.py) — data download, tokenizer training, dataloader, and evaluation metric (DO NOT MODIFY by agents).
- [train.py](train.py) — model, optimizer, hyperparameters, and training loop. This is the only file agents may edit during experiments.
- [program.md](program.md) — agent-facing instructions and experiment loop (human-maintained).
- [pyproject.toml](pyproject.toml) — runtime dependencies.

**High-level dataflow**
1. `prepare.py` downloads Parquet shards to `~/.cache/autoresearch/data/` and trains a BPE tokenizer saved to `~/.cache/autoresearch/tokenizer/`.
2. `train.py` loads the tokenizer (via `Tokenizer.from_directory()`), builds the GPT model, sets up optimizer, and creates a streaming dataloader from parquet shards (`make_dataloader`).
3. Training: the script runs a loop for a fixed wall-clock budget (`TIME_BUDGET = 300s`) and evaluates `val_bpb` using the fixed `evaluate_bpb` function in `prepare.py`.
4. Agent records results in `results.tsv` (untracked) with columns `commit, val_bpb, memory_gb, status, description`.

**Core components & responsibilities**
- **Tokenizer & data prep (`prepare.py`)**
  - Downloads shards, trains a Rust BPE tokenizer, saves a `tokenizer.pkl` (tiktoken Encoding) and `token_bytes.pt` used for BPB evaluation.
  - Provides `make_dataloader(tokenizer, B, T, split)` which packs documents into BOS-aligned batches with best-fit packing (no padding).
  - Provides `evaluate_bpb(model, tokenizer, batch_size)` (fixed metric for comparisons). Agents must not modify this evaluation.

- **Model (`train.py`)**
  - Implements a compact GPT-like model with per-layer residual scalars and optional value embeddings (ResFormer-style augmentation).
  - Rotary positional embeddings are precomputed and registered as buffers (cos, sin).
  - Uses a FlashAttention-3 backend via `kernels.get_kernel(...).flash_attn_interface`.

- **Optimizer**
  - `MuonAdamW` combines a Muon-style optimizer (for matrix-shaped params) with AdamW for other parameters.
  - A set of fused steps are compiled with `torch.compile` and custom scheduling (momentum, weight decay, LR multipliers).

- **Training loop**
  - Training is accumulation-based: `TOTAL_BATCH_SIZE` is achieved via gradient accumulation of micro-batches sized by `DEVICE_BATCH_SIZE * MAX_SEQ_LEN`.
  - Loop runs until `TIME_BUDGET` is reached (warmup/compile overhead excluded). After training, model is evaluated with `evaluate_bpb`.

**Agent guidance (what an agent should and should not do)**
- Can modify: `train.py` only. The agent can change architecture, hyperparameters, optimizer settings, and batch sizing. The intent is to explore improvements that reduce `val_bpb` under the fixed 5-minute budget.
- Must not modify: `prepare.py`, `evaluate_bpb`, or `program.md` (unless human authorizes changes). Do not add new dependencies (only use what's in `pyproject.toml`).
- Workflow for an experiment (as encoded in `program.md`):
  1. Create a fresh branch `autoresearch/<tag>`.
  2. Modify `train.py` and commit.
  3. Run `uv run train.py > run.log 2>&1`.
  4. Grep `val_bpb` and `peak_vram_mb` from `run.log` and append a row to `results.tsv` (untracked).
  5. If `val_bpb` improved, keep commit; otherwise reset.

**Developer notes & code pointers**
- Tokenizer: `Tokenizer` class in [prepare.py](prepare.py) wraps the tiktoken encoding; see `train.py` lines where `Tokenizer.from_directory()` is called.
- Dataloader: `make_dataloader` in [prepare.py](prepare.py) (best-fit packing, infinite iterator). Important for reproducible token counts.
- Evaluation: `evaluate_bpb(model, tokenizer, batch_size)` in [prepare.py](prepare.py) — uses `token_bytes.pt` to convert token-level cross entropy to bits/byte.
- Model internals: `GPT`, `Block`, `CausalSelfAttention`, and rotary embedding precomputation are in [train.py](train.py). Per-layer value embeddings live in `GPT.value_embeds`.
- Optimizer: `MuonAdamW` class and fused kernel steps live in [train.py](train.py) (search for `class MuonAdamW`). This code is single-GPU and relies on compiled fused ops.

**How to run (developer quick commands)**
1. Install dependencies per `pyproject.toml` (the project uses `uv` for environment management):

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
uv sync
```

2. Prepare data & tokenizer (one-time):

```bash
uv run prepare.py
```

3. Run a single training experiment (5 minutes):

```bash
uv run train.py
```

4. Extract results after run:

```bash
grep "^val_bpb:" run.log
```

**Recommended edits process for contributors**
- Keep changes scoped to `train.py` when prototyping experiments. Make small, incremental commits with clear descriptions saved in `results.tsv`.
- If you change data formats, tokenizer, or evaluation, update this documentation and notify collaborators — these are breaking changes for comparability.

**Maintenance & future improvements**
- Add a test harness to run a micro experiment in CI (CPU or small synthetic data) to catch syntax/runtime regressions.
- Consider splitting large optimizer code into a separate module for clarity and unit testing.
- Add automated logging to write `results.tsv` entries with a small CLI helper (so agents and humans use the same format).

**Appendix — quick code map**
- `prepare.py`
  - `download_data`, `train_tokenizer`, `Tokenizer.from_directory`, `make_dataloader`, `evaluate_bpb`.
- `train.py`
  - `GPTConfig`, `GPT`, `MuonAdamW`, LR/momentum schedules, main training loop.

---
If you'd like, I can also:
- add a small `docs/` folder with diagrams (Mermaid) showing dataflow, or
- create a tiny CLI helper to append standardized rows to `results.tsv`.

Requested-by: repository analysis task — generated documentation for agents and developers.
