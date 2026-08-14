# OpenLane-V2 adapter

The production adapter reads the OpenLane-V2 v2.1 Map Element Bucket `-ls.json` schema directly.
It yields identifiers in stable split, segment, and numeric timestamp order and parses one metadata file at a time.
It does not import or trust the upstream split-wide pickle produced by `preprocess-ls.py`.

## Normalized contract

Every adapted frame preserves the raw dataset version, source name, source segment identity, frame identity, source object IDs, and source traffic-element box coordinates.
Ground geometry is converted from x-right, y-forward, z-up into x-forward, y-left, z-up.
Camera extrinsics become `T_vehicle_camera`, and ego poses become `T_world_vehicle`.
Map Element Bucket lanes, boundaries, boundary types, connector flags, traffic controls, areas, lane-lane topology, and lane-traffic topology retain their source list order.
Categories, IDs, finite coordinates, topology values, topology shapes, rigid transforms, and image paths fail closed when malformed.

The camera contract always contains eight canonical slots.
Unavailable source cameras remain present with `valid=false`.
Source camera order is never used as tensor order.

## Lazy images and model tensors

Loading or iterating metadata never decodes an image.
The official metadata schema does not publish image width and height in each frame, so the adapter pins camera-specific dimensions from the owning source-dataset contracts.
The Argoverse 2 front-center camera is 1550 by 2048 pixels, while its other ring cameras are 2048 by 1550 pixels.
The nuScenes cameras are 1600 by 900 pixels.
An explicit source width and height in metadata overrides the source pin, which supports bounded repository-owned fixtures.

`OpenLaneAdapter.load_camera_rgb` decodes one requested camera and checks its observed dimensions against the normalized frame contract.
`OpenLaneAdapter.model_camera_inputs` materializes `[8, 3, 384, 640]` float32 RGB tensors, a Boolean camera mask, transformed intrinsics, and `T_vehicle_camera` tensors.
Invalid slots contain zeros.
Valid images use the recorded bilinear integer letterbox plan, ImageNet normalization values from the model profile, and immutable NumPy arrays.

## Official parity gate

The parity selector is `configs/data/openlane-v2-v2.1.parity.yaml`.
It contains public identifiers from the upstream sample manifest and no restricted annotations or images.
The parity runner mounts the registered dataset read-only into the locked Python 3.8 official-evaluator image, disables networking and Linux capabilities, and projects the selected frames through the pinned devkit `Frame` API.
The host adapter independently inverts canonical normalization and compares every selected metadata, calibration, pose, Map Element Bucket, and topology field with an absolute numeric tolerance of `1e-9`.
Only the selected frame count, maximum numeric error, devkit version, and projection-set hash are emitted.

Run the unrestricted local package gate with:

```sh
./tools/jl verify-m2-1-local
```

After acknowledging and registering the official sample, run the target data gate with:

```sh
uv run --locked junctionlens data verify-adapter --profile sample
```

The local package is `IMPLEMENTED_LOCAL` after the first command passes.
Milestone 2.1 becomes `ACCEPTED` only after the licensed official-devkit comparison also passes.

The image dimensions are pinned from the [Argoverse 2 Sensor guide](https://argoverse.github.io/user-guide/datasets/sensor.html) and the [nuScenes tutorial](https://www.nuscenes.org/public/tutorials/nuscenes_tutorial.html).
