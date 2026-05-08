# CREMI Affinity U-Net

CREMI-style affinity pipeline:

```
raw HDF EM -> 3D anisotropic U-Net -> long-range disaffinity maps -> watershed/agglomeration -> neuron instances
```

The active training, prediction, and segmentation scripts live at the repository root.

## Setup

`uv` is used for environment management:

```
uv sync
```

Run commands through `uv run` so they use the locked project environment.


The expected training data lives under `samples/train` and should contain CREMI HDF files with:

- `volumes/raw`
- `volumes/labels/neuron_ids`

The raw-only A+/B+/C+ files under `samples/test` are intended for inference.

You can also run the same commands through the root dispatcher, for example `python main.py train ...`.

## Train Disaffinities

Defaults follow the torch-em CREMI affinity baseline: five anisotropic scale levels, 12 long-range offsets, torch-em-style Dice loss, standardized raw input, EM defect augmentation, batch size 1, `initial_features=32`, and 3D crops of `32x360x360`.

```
uv run python train.py --train-dir samples/train --checkpoint-dir checkpoints/affinity
```

Larger experiment, more VRAM needed (~32GB recommended):

```
uv run python train.py --initial-features 64 --max-features 2048 --patch-shape 32 360 360
```

Test run (CPU-only):

```
uv run python train.py --steps 2 --val-every 2 --save-every 2 --patch-shape 16 144 144 --initial-features 4 --max-features 64 --num-workers 0 --device cpu
```

## Predict Maps

```
uv run python predict.py samples/test --checkpoint checkpoints/affinity/best.pth.tar --output-dir predictions/affinities
```

Full raw CREMI test volumes produce large 12-channel HDF outputs. New checkpoints write `target_mode=disaffinity` metadata so the segmenter can invert them for local agglomeration. Use `--block-shape` and `--halo` to tune inference memory.

## Segment Instances

```
uv run python segment.py predictions/affinities/sample_A+_padded_20160601_affinities.h5 --output-dir predictions/instances
```

The local segmentation backend is intended as a baseline. For very large volumes, start with `--roi z0 z1 y0 y1 x0 x1` or increase `--max-voxels` only if the machine has enough RAM.
