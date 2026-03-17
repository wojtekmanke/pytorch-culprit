# Changelog

## 0.1.0

The first version that does what the README says.

Methods, all behind `fit` and `score`:

- exact influence functions, as the reference;
- influence with the inverse applied by conjugate gradient, by LiSSA, by
  the top eigenpairs from Lanczos, and by K-FAC and EK-FAC;
- TracIn over checkpoints and the plain gradient dot product;
- TRAK, with a sparse random projection and a resumable gradient store;
- unrolled differentiation through a whole training run.

What they are built on:

- per-example gradients by `vmap` over `grad` and by forward hooks, and
  gradient dot products that do not form the gradients;
- Hessian-vector and Gauss-Newton-vector products, matrix-free operators,
  conjugate gradient and Lanczos;
- functional AdamW and SGD that match `torch.optim`, and many models
  trained at once through `vmap`;
- a fused RMS norm as a custom `autograd.Function` with hand-written
  `backward`, `jvp` and `vmap` rules;
- LoRA adapters as a parametrization, with every method working through
  them, and a wrapper for Hugging Face causal language models.

How they are checked:

- against dense linear algebra in float64 on models small enough for it;
- against retraining on random subsets, with the linear datamodeling score;
- against sources known by construction, in a planted-facts corpus and in a
  language model with a planted backdoor.

Known limits:

- Layers with parameters other than `Linear`, `Embedding`, `LayerNorm`,
  the fused `RMSNorm` and LoRA-adapted `Linear` are not supported by the
  hook-based methods, and a weight shared between two layers is refused.
- The exact Fisher is only available for one prediction per example.
- Everything was developed and measured on one machine, an Apple M5 Max;
  nothing has been run on CUDA.
