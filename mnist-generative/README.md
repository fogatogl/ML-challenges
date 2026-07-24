# mnist-generative

Implementing VAE, GAN, Diffusion (DDPM), and an Autoregressive Transformer
from their equations, on MNIST, on a single-GPU Onyxia/SSPCloud instance.

- **Plan:** [`LEARNING_PLAN.md`](./LEARNING_PLAN.md) — phases, milestones, deliverables.
- **Setup:** [`SETUP.md`](./SETUP.md) — one-time Onyxia onboarding (start here).
- **Log:** [`experiments/LOG.md`](./experiments/LOG.md) — one row per run, no exceptions.

```
src/            importable modules (data, models/, engine, metrics, viz)
scripts/        entry points: check_env.py, train_*.py
configs/        one yaml per experiment
experiments/    LOG.md
notebooks/      exploration only — real code lives in src/
```

Quick start on a fresh Onyxia service:
```bash
cd ~/work/ML-challenges/mnist-generative
python scripts/check_env.py
```

Lives inside the [`fogatogl/ML-challenges`](https://github.com/fogatogl/ML-challenges)
repo as a subfolder — see `SETUP.md` for how this connects to git/Onyxia.
