"""
Evaluation metrics: MNIST-FID, mode-coverage histogram, throughput.

TODO (Phase 0.3 depends on this file existing with a minimal API):
  - get_features(x) -> penultimate-layer activations of your Phase-0 classifier
  - predict(x)       -> class predictions, for the mode-collapse histogram
  - mnist_fid(real_feats, fake_feats) -> Frechet-distance-style score using
    your own classifier's features instead of ImageNet-Inception (which is
    meaningless for digits).

These two functions (get_features, predict) are the ✅ "done" criterion for
Milestone 0 in the learning plan — build them once the classifier hits 99%.
"""
