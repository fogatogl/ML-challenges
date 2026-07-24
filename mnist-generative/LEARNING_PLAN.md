# Generative Deep Learning on MNIST — Learning & Project Plan

> **Goal:** learn to *implement, train, debug, and evaluate* four families of generative models — VAE, GAN, Diffusion (DDPM), Autoregressive Transformer — in PyTorch on a single-GPU Onyxia/SSPCloud instance. Code-first: every model is written from its equations, not copied from a repo.

---

## How to work through this plan

1. **Phases are gated.** Don't start a phase until the previous milestone criteria are met.
2. **One change per experiment.** Fix seeds (`torch.manual_seed`, `numpy`, dataloader workers). Log everything in `experiments/LOG.md`: date, git commit, config, result, one-line conclusion.
3. **Implement from the math first.** Only after your version trains do you compare against a reference implementation, and you write down every difference you find.
4. **Notebooks are for exploration only.** All real code lives in `src/` as importable modules. This is half the point of the project: writing ML code like software, not like a script.
5. Each phase ends with **"questions you must be able to answer"** — if you can't, you're not done, even if the samples look fine.

---

## Phase 0 — Environment, infrastructure, baseline (2–4 days)

### 0.1 Onyxia / SSPCloud specifics — read this first

- **Services are ephemeral.** When your VSCode-pytorch service is deleted (or crashes), local files are gone. Non-negotiable rules:
  - Code lives in **Git**. Commit and push at least daily; ideally per experiment.
  - Datasets and checkpoints live in **S3 (MinIO)**, which SSPCloud injects credentials for. Use the `mc` client (preinstalled) or `s3fs`/`boto3` to `cp` checkpoints up after each run.
  - Put dependency installs in the service **init script** so a fresh instance is ready in one click.
- Sanity check on first launch:
  ```bash
  nvidia-smi
  python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0))"
  ```

### 0.2 Project skeleton

```
mnist-generative/
├── configs/            # one small yaml per experiment (lr, batch, arch dims…)
├── src/
│   ├── data.py         # dataloaders + normalization variants
│   ├── models/
│   │   ├── classifier.py
│   │   ├── vae.py
│   │   ├── gan.py
│   │   ├── unet.py     # for diffusion
│   │   └── transformer.py
│   ├── engine.py       # generic loop: logging, checkpoint/resume, EMA, sample hooks
│   ├── metrics.py      # MNIST-FID, per-class histogram, samples/sec
│   └── viz.py          # sample grids, interpolations, training GIFs
├── scripts/            # train_vae.py, train_gan.py, train_ddpm.py, train_ar.py
├── experiments/LOG.md
├── notebooks/
└── requirements.txt
```

**Normalization matters and differs per model** — build `data.py` to support both:
- `[0, 1]` pixels → VAE with Bernoulli decoder (BCE loss).
- `[-1, 1]` pixels → GAN and diffusion (tanh-range outputs).

### 0.3 Milestone 0 — CNN classifier ≥ 99% test accuracy

Why a classifier in a generative project? Three reasons:
1. Validates your entire pipeline (data, loop, GPU, logging) on an easy problem.
2. Its **penultimate-layer features** become your FID feature extractor (**MNIST-FID**) — real FID uses Inception on ImageNet, which is meaningless for digits.
3. Its **predictions on generated samples** give per-class counts → your mode-collapse detector.

**Exercises:** write the training loop by hand (no Lightning); add checkpoint save/resume mid-epoch; add TensorBoard logging; toggle mixed precision (`torch.autocast` + `GradScaler`) and measure the speedup.

✅ **Done when:** ≥99% test accuracy, a run can be killed and resumed, checkpoints land on S3, and you have a `get_features(x)` and `predict(x)` API in `metrics.py`.

---

## Phase 1 — VAE (week 1–2)

**Why first:** single stable loss, fast training, introduces latent-variable modeling — and it produces the encoder you'll reuse for evaluation intuition and (later) latent diffusion.

### Steps
1. **Plain autoencoder** (conv encoder → latent → conv decoder), reconstruction loss only. Look at reconstructions, then try sampling random latents — see why it fails as a generative model.
2. **Upgrade to VAE:** encoder outputs `mu, logvar`; reparameterization `z = mu + exp(0.5·logvar) ⊙ ε`; loss = reconstruction + KL, where for a diagonal Gaussian vs. N(0, I):
   `KL = -0.5 · Σ(1 + logvar − mu² − exp(logvar))`
3. **Experiments:**
   - `latent_dim ∈ {2, 16, 64}` — with `latent_dim=2`, scatter-plot the test-set latents colored by digit label. This picture is the whole point of VAEs.
   - **β-VAE:** `β ∈ {0.5, 1, 4}` — observe the reconstruction ↔ structure trade-off.
   - **KL annealing:** ramp β from 0→1 over the first epochs; compare against no warm-up.
   - Bernoulli (BCE) vs. Gaussian (MSE) decoder — what assumption does each encode about pixels?

### Classic bugs to hit on purpose
- **Reduction mismatch:** BCE summed over 784 pixels vs. KL summed over `latent_dim` dims — if you `mean` one and `sum` the other, the balance silently breaks. Get this wrong once, diagnose it from the loss curves, then fix it.
- **Posterior collapse:** KL → 0, decoder ignores z, all samples look identical/average.

### Deliverables
2D latent scatter plot · interpolation grid between two digits · grid of samples from the prior · short note: *why are VAE samples blurry?*

✅ **Done when:** prior samples are recognizable digits, and you can answer: What exactly does the reparameterization trick make possible, and why can't you backprop through `torch.randn`-then-select? Why does β>1 blur but organize the latent space? What happens to the ELBO if you double `latent_dim`?

---

## Phase 2 — GAN (week 2–3)

**Why second:** you now meet *unstable* training. The lessons here are diagnostic skills, not architecture.

### Steps
1. **MLP GAN** on flattened digits with the **non-saturating loss** (G maximizes `log D(G(z))`). Expect ugliness and instability — that's the data point.
2. **DCGAN:** conv G and D, BatchNorm in G, LeakyReLU(0.2) in D, `Adam(lr=2e-4, betas=(0.5, 0.999))`, tanh output on `[-1,1]` data. Keep a **fixed noise grid** and save the generated images every N steps → training-evolution GIF.
3. **Stabilization ablations** (one at a time): one-sided label smoothing (real=0.9) · different lr for D and G (TTUR) · spectral norm on D.
4. **Conditional GAN:** feed the class label to both G and D (embedding + concat or projection). Now you can *ask* for a 7.
5. *(Optional)* **WGAN-GP** — a different divergence, and the D loss finally correlates with sample quality.

### Diagnostics to build (this is the real skill)
- **Mode collapse detector:** generate 10k samples, classify with your Phase-0 classifier, plot the class histogram. A healthy G is ~uniform.
- Read the curves: `D_loss → 0` with garbage samples = D overpowered; oscillating losses that never settle = lr/balance issue. Keep a written table: *symptom → likely cause → fix that worked*.

### Deliverables
Fixed-noise training GIF · class histogram before/after fixes · conditional grid (one row per digit) · your symptom→fix table.

✅ **Done when:** conditional DCGAN produces sharp, class-correct digits with a roughly uniform unconditional histogram, and you can answer: Why does the original minimax G loss saturate early in training? Why sharp samples but mode-dropping (contrast with the VAE's Gaussian likelihood)? What does BatchNorm in D do to the independence assumption between samples?

---

## Phase 3 — Diffusion / DDPM (week 3–5)

**Why third:** the current state-of-the-art family. Conceptually heavier, but training is a *stable regression* — a relief after GANs, and the contrast is instructive.

### Steps
1. **Forward process first, no network.** Implement the closed form
   `x_t = √(ᾱ_t)·x_0 + √(1−ᾱ_t)·ε` and visualize a digit at t = 0, 50, 100, …, 1000 for a **linear** and a **cosine** β-schedule. Understand what "the schedule" destroys and when.
2. **Small U-Net** with sinusoidal **time embeddings** (t → sinusoid → MLP → added to feature maps per block). Start attention-free; MNIST doesn't need attention to work.
3. **Training loop is 5 lines:** sample `t`, sample `ε`, build `x_t`, predict `ε̂ = ε_θ(x_t, t)`, MSE. Add an **EMA copy of the weights** (decay ≈ 0.999) — sample from the EMA model, it's a large free quality win.
4. **Sampling:** ancestral DDPM loop (1000 steps), then **DDIM** to sample in 25–50 steps. Measure the wall-clock difference and the quality difference.
5. **Classifier-free guidance (CFG):** train with the label dropped ~10% of the time; at sampling use
   `ε̃ = ε_θ(x_t, ∅) + s·(ε_θ(x_t, c) − ε_θ(x_t, ∅))` and sweep `s ∈ {1, 2, 4, 8}`.

### Experiments
Linear vs. cosine schedule (MNIST-FID) · T = 200 vs. 1000 · DDIM steps ∈ {10, 25, 50} vs. quality · CFG scale grid (watch quality go up, then diversity collapse).

### Deliverables
Forward-corruption strip · reverse-denoising strip (x_T → x_0) · CFG scale comparison grid · FID-vs-sampling-steps plot.

✅ **Done when:** EMA + DDIM gives clean conditional digits in ≤50 steps, and you can answer: Why predict ε instead of x₀ directly (what happens to loss weighting across t)? Why is diffusion training stable where GAN training isn't? What is CFG doing geometrically to the score?

---

## Phase 4 — Autoregressive Transformer (week 5–6)

**Why last:** images-as-sequences bridges you to modern LLM-style modeling, and it forces the question every AR model faces: *what is a token?*

### Steps
1. **Tokenize MNIST:** quantize pixels to 16 gray levels → vocab of 16, sequence of 28×28 = 784 tokens (or downsample to 14×14 = 196 tokens to iterate faster).
2. **Write causal self-attention from scratch** — QKV projections, mask, multi-head — before ever touching `nn.Transformer`. Then a GPT-style stack: token emb + positional emb, ~4 layers, d_model 128–256, 4 heads.
3. **Train:** next-token cross-entropy, teacher forcing, cosine LR schedule with warmup. Track loss in **bits per pixel** — this family gives you an *exact* likelihood, use it.
4. **Sample:** temperature and top-k. Then the showpiece — **completion**: condition on the top half of a real test digit, generate the bottom half.
5. *(Stretch, highly recommended)* **VQ-VAE + transformer:** train a VQ-VAE (codebook, commitment loss, straight-through estimator) compressing 28×28 → 7×7 discrete codes, then run the transformer over 49 tokens. This is the DALL·E-1 / latent-space recipe in miniature and reuses your Phase-1 skills.

### Deliverables
Unconditional samples at 2–3 temperatures · half-image completion grid · bits-per-pixel on the test set · note comparing raw-pixel vs. VQ tokenization cost.

✅ **Done when:** completions are coherent, and you can answer: Why is the causal mask equivalent to the chain-rule factorization of p(x)? Why is AR sampling O(sequence length) and what do KV caches change? Why does tokenization (VQ) matter more than model size for images?

---

## Phase 5 — Unified evaluation & write-up (week 6–7)

Freeze one "best" model per family and evaluate identically:

| Metric | How |
|---|---|
| MNIST-FID | classifier features, 10k generated vs. 10k test |
| Mode coverage | class histogram entropy from your classifier |
| Sample quality | fixed grid, same layout for all four |
| Sampling speed | samples/sec, batch=64 |
| Training cost | wall-clock + steps to reach FID ≤ threshold |
| Params | count |
| Likelihood | bits/pixel where it exists (VAE bound, AR exact) |

Write a 2–3 page report: methods, the comparison table, and *which model you'd pick for three concrete scenarios* (see cheat sheet below). This document is the portfolio artifact.

---

## Phase 6 — Extensions (open-ended)

- **Fashion-MNIST** (drop-in), then **CIFAR-10** (color + real structure — everything gets harder; diffusion will need attention blocks and more channels).
- **Latent diffusion:** run your DDPM inside your Phase-1 VAE latent space — the Stable-Diffusion idea, built entirely from your own parts.
- **Conditioning everywhere:** make all four models class-conditional and compare controllability.
- `torch.compile` your training steps and measure.

---

## Choosing a model — decision cheat sheet

| Criterion | VAE | GAN | Diffusion | AR Transformer |
|---|---|---|---|---|
| Sample quality | blurry | sharp | best | good (needs good tokens) |
| Mode coverage / diversity | good | collapse risk | excellent | excellent |
| Training stability | high | **low** | high | high |
| Sampling speed | 1 pass | 1 pass | many passes | sequential (slow) |
| Explicit latent space | yes, smooth | implicit only | noise space | none |
| Likelihood | ELBO (lower bound) | none | bound | **exact** |
| Compute to train (MNIST) | minutes | ~1 h | hours | ~1–3 h |

**Rules of thumb:**
- Need **representations** (anomaly detection, compression, downstream features, interpolation) → **VAE**.
- Need **fast inference** of sharp samples, or image-to-image style tasks → **GAN**.
- Need **best quality + diversity + controllability**, can pay at sampling time → **diffusion**.
- Data is **discrete/sequential** (text, code, music, tokenized images), or you need exact likelihoods → **AR transformer**.
- Modern systems **compose** them: VQ-VAE/VAE for tokens or latents + diffusion or AR on top.

---

## Getting the most from limited compute (single GPU)

**Ladder of interventions — cheapest first:**
1. **Pipeline:** `num_workers>0`, `pin_memory=True`, biggest batch that fits, mixed precision (often ~2× on modern GPUs), gradient accumulation if a paper batch size doesn't fit.
2. **Right-size the model:** MNIST needs ≲5M params per model. Bigger mostly wastes your GPU hours here.
3. **Optimization hygiene:** EMA weights (diffusion & GAN generator), cosine schedule + warmup (transformer & diffusion), correct Adam betas per family.
4. **Family-specific stabilizers:** KL annealing (VAE) · spectral norm / TTUR / label smoothing (GAN) · cosine noise schedule + CFG (diffusion) · dropout + weight decay (transformer).
5. **Scale (params, steps) last** — and only when the evaluation metric, not vibes, says the model is the bottleneck.

**Caution on data augmentation:** in generative modeling, augmentations change the distribution you are learning (rotate MNIST and your model generates rotated digits). Keep augmentation off by default; the exception is discriminator-only augmentation for GANs (DiffAugment-style).

**Checkpoint discipline:** save every N minutes *and* at every eval; push to S3; store the config and git hash inside the checkpoint dict.

---

## Suggested timeline (part-time)

| Week | Focus |
|---|---|
| 1 | Phase 0 + start VAE |
| 2 | Finish VAE + start GAN |
| 3 | GAN stabilization + conditional |
| 4–5 | Diffusion |
| 5–6 | Transformer |
| 7 | Evaluation + report |
| 8+ | Extensions |
