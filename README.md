# Salience-Gated Slot Memory for State-Space Models (SG-SSM)
============================================================

Core algorithm: a bounded, content-addressable slot pool that runs alongside
a standard (fixed-rank) SSM recurrence. Rare/high-salience tokens get written
into a dedicated slot (near-lossless, independently-decaying) instead of being
superimposed into the shared low-rank state. At read time, a gate blends a
sparse cross-attention readout over the slot pool with the normal SSM readout.

This file contains:
  1. NaiveSSM        - a minimal, diagonal, input-selective SSM (Mamba-style
                        selection, S4D-style diagonal recurrence). Pure
                        PyTorch, sequential scan -> intentionally simple and
                        CPU-runnable, NOT a performance-optimized kernel.
  2. SlotPool         - the salience-gated associative memory.
  3. SGSSMLayer       - base SSM + slot pool + gated readout, combined.
  4. mqar_batch()      - synthetic Multi-Query Associative Recall task
                        generator (Zoology/Based-style stress test).
  5. train_and_eval()  - trains a baseline vs. SG-SSM model on MQAR across
                        varying (num_pairs, gap) and reports accuracy.

Everything here is intentionally small-scale: this is a proof-of-concept
sized for a laptop/CPU, meant to demonstrate the *mechanism*, not a
production kernel. Scaling it up (real Mamba CUDA kernels, real language
data) is future work -- see the paper draft.
