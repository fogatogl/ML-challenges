"""
Generic training loop utilities shared across all four model families.

TODO: as you build each model's train script, extract the common pieces here:
  - training loop skeleton (step, log, eval, checkpoint)
  - checkpoint save/resume (store config + git commit hash in the dict)
  - EMA weight tracking (needed for diffusion + GAN generator)
  - simple TensorBoard logging wrapper
  - sample-generation hooks (call every N steps, save to disk / S3)

Don't over-engineer this on day one — start by copy-pasting between
train_vae.py / train_gan.py, and only pull the shared parts in here once
you see the duplication with your own eyes.
"""
