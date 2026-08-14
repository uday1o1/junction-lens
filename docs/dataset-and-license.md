# Dataset and license boundary

JunctionLens source code is licensed under Apache License 2.0.

The OpenLane-V2 devkit is a separate Apache-2.0 work pinned by `configs/data/openlane-v2-v2.1.lock.yaml`.
The OpenLane-V2 dataset is distributed separately under CC BY-NC-SA 4.0.
OpenLane-V2 also requires agreement to the upstream nuScenes and Argoverse 2 terms before use.

JunctionLens does not automatically download or redistribute OpenLane-V2 images, annotations, preprocessed pickles, dataset-derived thumbnails, or trained weights.
A user must acknowledge all named terms through the local data-registration workflow before a licensed dataset root can be registered.
The acknowledgment receipt is machine-local and ignored by Git.
The immutable dataset-registration artifact stores a redacted receipt hash, not personal data or a machine path.

The unrestricted `synthetic` profile uses only repository-owned generated geometry, controls, calibration, and renderings.
CI and the public demonstration use that profile.
