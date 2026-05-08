from __future__ import annotations

import argparse
from pathlib import Path

import h5py
import numpy as np
from scipy import ndimage as ndi
from skimage.segmentation import watershed

from config import DEFAULT_OFFSETS
from data import slices_for_offset


class UnionFind:
    def __init__(self, size: int):
        self.parent = np.arange(size, dtype=np.int64)

    def find(self, value: int) -> int:
        parent = self.parent
        root = value
        while parent[root] != root:
            root = int(parent[root])
        while parent[value] != value:
            next_value = int(parent[value])
            parent[value] = root
            value = next_value
        return root

    def union(self, a: int, b: int) -> None:
        root_a = self.find(a)
        root_b = self.find(b)
        if root_a != root_b:
            self.parent[root_b] = root_a


def parse_roi(values: list[int] | None) -> tuple[slice, slice, slice] | None:
    if values is None:
        return None
    if len(values) != 6:
        raise argparse.ArgumentTypeError("ROI must be z0 z1 y0 y1 x0 x1")
    z0, z1, y0, y1, x0, x1 = values
    return slice(z0, z1), slice(y0, y1), slice(x0, x1)


def _decode_attr(value) -> str:
    if isinstance(value, bytes):
        return value.decode("utf-8")
    return str(value)


def load_affinities(
    path: Path,
    key: str,
    roi: tuple[slice, slice, slice] | None,
    target_mode_override: str | None,
) -> tuple[np.ndarray, tuple[tuple[int, int, int], ...], str]:
    with h5py.File(path, "r") as handle:
        if key not in handle:
            raise KeyError(f"{path} does not contain {key}")
        dataset = handle[key]
        offsets = tuple(tuple(int(v) for v in offset) for offset in dataset.attrs.get("offsets", DEFAULT_OFFSETS))
        target_mode = target_mode_override or _decode_attr(dataset.attrs.get("target_mode", "affinity"))
        if roi is None:
            affinities = np.asarray(dataset)
        else:
            affinities = np.asarray(dataset[(slice(None), *roi)])

    affinities = affinities.astype(np.float32, copy=False)
    if affinities.max(initial=0.0) > 1.0:
        affinities /= 255.0
    if target_mode == "disaffinity":
        affinities = 1.0 - affinities
    elif target_mode != "affinity":
        raise ValueError(f'Unknown target_mode "{target_mode}". Expected affinity or disaffinity.')
    return affinities, offsets, target_mode


def make_fragments(
    affinities: np.ndarray,
    foreground_threshold: float,
    seed_distance: int,
) -> np.ndarray:
    direct_affinities = affinities[:3]
    mean_affinity = direct_affinities.mean(axis=0)
    foreground = mean_affinity >= foreground_threshold
    if not foreground.any():
        return np.zeros(mean_affinity.shape, dtype=np.uint64)

    distance = ndi.distance_transform_edt(foreground)
    local_max = distance == ndi.maximum_filter(distance, size=seed_distance)
    local_max &= foreground
    markers, marker_count = ndi.label(local_max)
    if marker_count == 0:
        markers, _ = ndi.label(foreground)
    fragments = watershed(1.0 - mean_affinity, markers=markers, mask=foreground)
    return fragments.astype(np.uint64, copy=False)


def aggregate_edges(
    fragments: np.ndarray,
    affinities: np.ndarray,
    offsets: tuple[tuple[int, int, int], ...],
    channels: int | None = None,
) -> dict[tuple[int, int], tuple[float, int]]:
    edge_stats: dict[tuple[int, int], tuple[float, int]] = {}
    shape = tuple(fragments.shape)
    channels_to_use = len(offsets) if channels is None else channels
    for channel, offset in enumerate(offsets[:channels_to_use]):
        if any(abs(off) >= size for off, size in zip(offset, shape)):
            continue
        source_slices, neighbor_slices = slices_for_offset(offset, shape)
        source = fragments[source_slices]
        neighbor = fragments[neighbor_slices]
        boundary = (source > 0) & (neighbor > 0) & (source != neighbor)
        if not boundary.any():
            continue

        left = source[boundary].astype(np.int64, copy=False)
        right = neighbor[boundary].astype(np.int64, copy=False)
        values = affinities[channel][source_slices][boundary].astype(np.float64, copy=False)
        pairs = np.stack((np.minimum(left, right), np.maximum(left, right)), axis=1)
        unique_pairs, inverse = np.unique(pairs, axis=0, return_inverse=True)
        sums = np.bincount(inverse, weights=values)
        counts = np.bincount(inverse)

        for pair, value_sum, count in zip(unique_pairs, sums, counts):
            key = int(pair[0]), int(pair[1])
            previous_sum, previous_count = edge_stats.get(key, (0.0, 0))
            edge_stats[key] = previous_sum + float(value_sum), previous_count + int(count)
    return edge_stats


def agglomerate_fragments(
    fragments: np.ndarray,
    affinities: np.ndarray,
    offsets: tuple[tuple[int, int, int], ...],
    merge_threshold: float,
    min_size: int,
) -> np.ndarray:
    max_label = int(fragments.max(initial=0))
    if max_label == 0:
        return fragments.astype(np.uint64, copy=False)

    uf = UnionFind(max_label + 1)
    edge_stats = aggregate_edges(fragments, affinities, offsets)
    for (a, b), (value_sum, count) in edge_stats.items():
        if count > 0 and value_sum / count >= merge_threshold:
            uf.union(a, b)

    roots = np.arange(max_label + 1, dtype=np.int64)
    for label in range(1, max_label + 1):
        roots[label] = uf.find(label)

    merged = roots[fragments]
    counts = np.bincount(merged.ravel())
    keep = counts >= min_size
    keep[0] = False
    merged[~keep[merged]] = 0

    unique = np.unique(merged)
    relabel = np.zeros(int(unique.max(initial=0)) + 1, dtype=np.uint64)
    relabel[unique[unique > 0]] = np.arange(1, int((unique > 0).sum()) + 1, dtype=np.uint64)
    return relabel[merged]


def save_segmentation(path: Path, output_key: str, segmentation: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(path, "w") as handle:
        handle.create_dataset(output_key, data=segmentation.astype(np.uint64, copy=False), compression="gzip", compression_opts=4)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Convert predicted affinities to neuron instance labels.")
    parser.add_argument("input", type=Path, help="Affinity HDF file.")
    parser.add_argument("--input-key", default="affinities")
    parser.add_argument("--target-mode", choices=("auto", "affinity", "disaffinity"), default="auto")
    parser.add_argument("--output-dir", type=Path, default=Path("predictions/instances"))
    parser.add_argument("--output-key", default="volumes/labels/neuron_ids")
    parser.add_argument("--foreground-threshold", type=float, default=0.5)
    parser.add_argument("--merge-threshold", type=float, default=0.65)
    parser.add_argument("--seed-distance", type=int, default=8)
    parser.add_argument("--min-size", type=int, default=64)
    parser.add_argument("--roi", nargs=6, type=int, default=None, metavar=("Z0", "Z1", "Y0", "Y1", "X0", "X1"))
    parser.add_argument("--max-voxels", type=int, default=150_000_000)
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    if not args.input.exists():
        raise FileNotFoundError(f"Input path not found: {args.input}")

    roi = parse_roi(args.roi)
    target_mode_override = None if args.target_mode == "auto" else args.target_mode
    affinities, offsets, target_mode = load_affinities(args.input, args.input_key, roi, target_mode_override)
    voxels = int(np.prod(affinities.shape[1:]))
    if voxels > args.max_voxels:
        raise MemoryError(
            f"Requested segmentation has {voxels:,} voxels, above --max-voxels={args.max_voxels:,}. "
            "Use --roi for a smaller region or raise --max-voxels if this machine has enough RAM."
        )

    fragments = make_fragments(affinities, args.foreground_threshold, args.seed_distance)
    segmentation = agglomerate_fragments(fragments, affinities, offsets, args.merge_threshold, args.min_size)
    output_path = args.output_dir / f"{args.input.stem}_instances.h5"
    save_segmentation(output_path, args.output_key, segmentation)
    print(f"Saved {output_path}")


if __name__ == "__main__":
    main()
