# Dataset and license boundary

JunctionLens source code is licensed under Apache License 2.0.

The OpenLane-V2 devkit is a separate Apache-2.0 work pinned by `configs/data/openlane-v2-v2.1.lock.yaml`.
The OpenLane-V2 dataset is distributed separately under CC BY-NC-SA 4.0.
OpenLane-V2 also requires agreement to the upstream nuScenes and Argoverse 2 terms before use.

JunctionLens does not automatically download or redistribute OpenLane-V2 images, annotations, preprocessed pickles, dataset-derived thumbnails, or trained weights.
A user must acknowledge all named terms through the local data-registration workflow before a licensed dataset root can be registered.
The acknowledgment receipt is machine-local and ignored by Git.
The machine-local registration receipt stores the canonical root and content hashes below ignored owner-only state.
Later immutable registry artifacts store only a redacted receipt hash, not personal data or a machine path.

The unrestricted `synthetic` profile uses only repository-owned generated geometry, controls, calibration, and renderings.
CI and the public demonstration use that profile.

## Licensed local workflow

Review the upstream terms before recording an acknowledgment.
JunctionLens does not accept terms on a user's behalf.

```sh
uv run --locked junctionlens data acknowledge \
  --accept-term CC-BY-NC-SA-4.0 \
  --accept-term nuScenes-terms \
  --accept-term Argoverse-2-terms \
  --confirm-restricted-noncommercial-use
```

Retain the original official archive so registration can verify the published MD5 before trusting the extracted root.
The sample root must contain the official `data_dict_example.json` manifest and the raw frame hierarchy.

```sh
uv run --locked junctionlens data register \
  --profile sample \
  --root /path/to/OpenLane-V2 \
  --archive /path/to/OpenLane-V2_sample.tar

uv run --locked junctionlens data audit --profile sample

uv run --locked junctionlens data verify-adapter --profile sample

uv run --locked junctionlens data manifest --profile full
```

Registration fails closed for a missing acknowledgment, checksum mismatch, missing manifest, changed root, or unsafe receipt path.
The audit loads raw `-ls.json` files directly and never accepts an upstream pickle across the application trust boundary.
It reports query-capacity coverage and separate lane, traffic-control, and road-area identity evidence.
The adapter parity command compares the frozen sample selector with the pinned official v2.1 devkit inside the locked compatibility image.
It emits hashes and aggregate numeric error only, so restricted annotations do not become repository artifacts.
The full-profile manifest command streams bounded provenance into the ignored content-addressed artifact store.
The [data manifests and V1 splits](data-manifests-and-splits.md) workflow freezes the exact learning partitions without exposing annotation or image content.

## Frozen normalization

The source frame uses x-right, y-forward, z-up coordinates.
The JunctionLens vehicle and world frame uses x-forward, y-left, z-up coordinates.
Extrinsics are normalized to `T_vehicle_camera`, and ego poses are normalized to `T_world_vehicle`.
Every frame has the same eight ordered camera slots, with unavailable source cameras represented explicitly as invalid slots.
Original traffic-element integer pixel boxes are retained separately from normalized half-open model boxes.
