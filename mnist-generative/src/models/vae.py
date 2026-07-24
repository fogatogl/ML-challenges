"""
Phase 1 — VAE.

Implementation order (see LEARNING_PLAN.md Phase 1 for the full spec):
  1. plain autoencoder (recon loss only)
  2. add mu/logvar heads + reparameterization trick
  3. add the KL term, watch the reduction (mean vs sum) match the recon loss
"""
