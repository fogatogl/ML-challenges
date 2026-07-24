"""
Dataloaders + normalization variants for MNIST.

TODO (Phase 0.2 / 0.3): implement dataloaders supporting two normalization
modes, since different model families expect different pixel ranges:
  - "unit"      -> pixels in [0, 1], for the VAE's Bernoulli decoder (BCE loss)
  - "tanh"      -> pixels in [-1, 1], for GAN / diffusion (tanh-range outputs)

Suggested API (fill in yourself — this is the first real exercise):

    def get_dataloaders(batch_size: int, normalization: str = "unit",
                         num_workers: int = 4) -> tuple[DataLoader, DataLoader]:
        ...

Remember: num_workers > 0, pin_memory=True, and NO data augmentation by
default (augmentation changes the distribution a generative model learns).
"""
