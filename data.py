from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset, get_worker_info

from config import DEFAULT_NORMALIZATION, DEFAULT_OFFSETS, DEFAULT_TARGET_MODE, LABEL_DATASET_KEY, RAW_DATASET_KEY


HDF_EXTENSIONS = {".h5", ".hdf", ".hdf5"}


@dataclass(frozen=True)
class CremiVolumeInfo:
    path: Path
    raw_key: str
    label_key: str | None
    raw_start: tuple[int, int, int]
    label_shape: tuple[int, int, int] | None
    raw_shape: tuple[int, int, int]


def iter_hdf_paths(path: Path) -> list[Path]:
    if path.is_file():
        if path.suffix.lower() not in HDF_EXTENSIONS:
            raise ValueError(f"Expected an HDF5 file, got {path}")
        return [path]

    paths = sorted(candidate for candidate in path.rglob("*") if candidate.suffix.lower() in HDF_EXTENSIONS)
    if not paths:
        raise FileNotFoundError(f"No HDF5 files found under {path}")
    return paths


def available_datasets(group: h5py.Group) -> str:
    names: list[str] = []

    def collect(name: str, obj: h5py.Dataset) -> None:
        if isinstance(obj, h5py.Dataset):
            names.append(name)

    group.visititems(collect)
    return ", ".join(names)


def _attr_vector(dataset: h5py.Dataset, name: str, default: Iterable[float]) -> np.ndarray:
    return np.asarray(dataset.attrs.get(name, default), dtype=float)


def aligned_raw_start(raw_dataset: h5py.Dataset, label_dataset: h5py.Dataset) -> tuple[int, int, int]:
    label_shape = label_dataset.shape
    raw_shape = raw_dataset.shape
    label_offset = _attr_vector(label_dataset, "offset", np.zeros(3))
    raw_offset = _attr_vector(raw_dataset, "offset", np.zeros(3))
    raw_resolution = _attr_vector(raw_dataset, "resolution", np.ones(3))
    label_resolution = _attr_vector(label_dataset, "resolution", raw_resolution)

    if not np.allclose(raw_resolution, label_resolution):
        raise ValueError(
            f"Raw and label resolutions differ: raw={raw_resolution}, labels={label_resolution}. "
            "The affinity loader expects matching voxel grids."
        )

    start = np.rint((label_offset - raw_offset) / raw_resolution).astype(int)
    z_start, y_start, x_start = start.tolist()
    z_count, height, width = label_shape
    fits_raw = (
        z_start >= 0
        and y_start >= 0
        and x_start >= 0
        and z_start + z_count <= raw_shape[0]
        and y_start + height <= raw_shape[1]
        and x_start + width <= raw_shape[2]
    )
    if fits_raw:
        return z_start, y_start, x_start
    if raw_shape[:3] == label_shape[:3]:
        return 0, 0, 0
    raise ValueError(
        f"Cannot align raw shape {raw_shape} with label shape {label_shape}. "
        f"Computed raw start index was {(z_start, y_start, x_start)}."
    )


def load_cremi_volume_infos(
    path: Path,
    raw_key: str = RAW_DATASET_KEY,
    label_key: str = LABEL_DATASET_KEY,
    require_labels: bool = True,
) -> list[CremiVolumeInfo]:
    infos: list[CremiVolumeInfo] = []
    for hdf_path in iter_hdf_paths(path):
        with h5py.File(hdf_path, "r") as handle:
            if raw_key not in handle:
                raise KeyError(f"{hdf_path} does not contain {raw_key}. Available: {available_datasets(handle)}")

            raw_dataset = handle[raw_key]
            if not isinstance(raw_dataset, h5py.Dataset):
                raise TypeError(f"{raw_key} in {hdf_path} is not an HDF5 dataset")

            if label_key not in handle:
                if require_labels:
                    raise KeyError(
                        f"{hdf_path} does not contain {label_key}. Available: {available_datasets(handle)}"
                    )
                infos.append(
                    CremiVolumeInfo(
                        path=hdf_path,
                        raw_key=raw_key,
                        label_key=None,
                        raw_start=(0, 0, 0),
                        label_shape=None,
                        raw_shape=tuple(raw_dataset.shape[:3]),
                    )
                )
                continue

            label_dataset = handle[label_key]
            if not isinstance(label_dataset, h5py.Dataset):
                raise TypeError(f"{label_key} in {hdf_path} is not an HDF5 dataset")
            infos.append(
                CremiVolumeInfo(
                    path=hdf_path,
                    raw_key=raw_key,
                    label_key=label_key,
                    raw_start=aligned_raw_start(raw_dataset, label_dataset),
                    label_shape=tuple(label_dataset.shape[:3]),
                    raw_shape=tuple(raw_dataset.shape[:3]),
                )
            )
    return infos


def normalize_raw(raw: np.ndarray, mode: str = DEFAULT_NORMALIZATION) -> np.ndarray:
    raw = raw.astype(np.float32, copy=False)
    if mode == "scale01":
        if raw.max(initial=0.0) > 1.0:
            raw = raw / 255.0
        return np.clip(raw, 0.0, 1.0)
    if mode == "standardize":
        mean = float(raw.mean())
        std = float(raw.std())
        return (raw - mean) / max(std, 1e-6)
    raise ValueError(f"Unknown normalization mode: {mode}")


def slices_for_offset(offset: tuple[int, int, int], shape: tuple[int, int, int]) -> tuple[tuple[slice, ...], tuple[slice, ...]]:
    source_slices = []
    neighbor_slices = []
    for off, size in zip(offset, shape):
        if off < 0:
            source_slices.append(slice(-off, size))
            neighbor_slices.append(slice(0, size + off))
        elif off > 0:
            source_slices.append(slice(0, size - off))
            neighbor_slices.append(slice(off, size))
        else:
            source_slices.append(slice(0, size))
            neighbor_slices.append(slice(0, size))
    return tuple(source_slices), tuple(neighbor_slices)


def labels_to_affinities(
    labels: np.ndarray,
    offsets: tuple[tuple[int, int, int], ...] = DEFAULT_OFFSETS,
    ignore_label: int = 0,
    target_mode: str = DEFAULT_TARGET_MODE,
) -> tuple[np.ndarray, np.ndarray]:
    if target_mode not in {"affinity", "disaffinity"}:
        raise ValueError('target_mode must be "affinity" or "disaffinity"')

    shape = tuple(labels.shape)
    targets = np.zeros((len(offsets), *shape), dtype=np.float32)
    mask = np.zeros_like(targets, dtype=np.float32)

    for channel, offset in enumerate(offsets):
        if any(abs(off) >= size for off, size in zip(offset, shape)):
            continue
        source_slices, neighbor_slices = slices_for_offset(offset, shape)
        source = labels[source_slices]
        neighbor = labels[neighbor_slices]
        valid = (source != ignore_label) & (neighbor != ignore_label)
        channel_targets = targets[channel][source_slices]
        channel_mask = mask[channel][source_slices]
        if target_mode == "affinity":
            channel_targets[valid] = source[valid] == neighbor[valid]
        else:
            channel_targets[valid] = source[valid] != neighbor[valid]
        channel_mask[valid] = 1.0

    return targets, mask


class CremiAffinityDataset(Dataset):
    def __init__(
        self,
        data_path: Path,
        patch_shape: tuple[int, int, int],
        raw_key: str = RAW_DATASET_KEY,
        label_key: str = LABEL_DATASET_KEY,
        offsets: tuple[tuple[int, int, int], ...] = DEFAULT_OFFSETS,
        patches_per_epoch: int = 1024,
        augment: bool = True,
        deterministic: bool = False,
        seed: int = 17,
        normalize: str = DEFAULT_NORMALIZATION,
        target_mode: str = DEFAULT_TARGET_MODE,
        p_drop_slice: float = 0.025,
        p_low_contrast: float = 0.025,
        p_deform_slice: float = 0.0,
    ):
        self.volumes = load_cremi_volume_infos(data_path, raw_key=raw_key, label_key=label_key, require_labels=True)
        self.patch_shape = tuple(patch_shape)
        self.offsets = tuple(tuple(offset) for offset in offsets)
        self.patches_per_epoch = patches_per_epoch
        self.augment = augment
        self.deterministic = deterministic
        self.seed = seed
        self.normalize = normalize
        self.target_mode = target_mode
        self.p_drop_slice = p_drop_slice
        self.p_low_contrast = p_low_contrast
        self.p_deform_slice = p_deform_slice
        self._handles: dict[Path, h5py.File] = {}
        self._rng: np.random.Generator | None = None

        for volume in self.volumes:
            if volume.label_shape is None:
                raise ValueError(f"{volume.path} does not have labels")
            if any(patch > size for patch, size in zip(self.patch_shape, volume.label_shape)):
                raise ValueError(
                    f"Patch shape {self.patch_shape} does not fit label shape {volume.label_shape} in {volume.path}"
                )

    def __len__(self) -> int:
        return self.patches_per_epoch

    def __getstate__(self) -> dict:
        state = self.__dict__.copy()
        state["_handles"] = {}
        state["_rng"] = None
        return state

    def _handle(self, path: Path) -> h5py.File:
        handle = self._handles.get(path)
        if handle is None:
            handle = h5py.File(path, "r")
            self._handles[path] = handle
        return handle

    def _rng_for_index(self, index: int) -> np.random.Generator:
        if self.deterministic:
            return np.random.default_rng(self.seed + index)

        if self._rng is None:
            worker = get_worker_info()
            worker_id = 0 if worker is None else worker.id
            self._rng = np.random.default_rng(self.seed + worker_id * 100_003)
        return self._rng

    def _sample_patch_origin(
        self, rng: np.random.Generator
    ) -> tuple[CremiVolumeInfo, tuple[int, int, int]]:
        volume = self.volumes[int(rng.integers(0, len(self.volumes)))]
        assert volume.label_shape is not None
        origin = tuple(
            int(rng.integers(0, size - patch + 1)) for size, patch in zip(volume.label_shape, self.patch_shape)
        )
        return volume, origin

    def _augment_spatial(
        self, raw: np.ndarray, labels: np.ndarray, rng: np.random.Generator
    ) -> tuple[np.ndarray, np.ndarray]:
        if rng.random() < 0.5:
            raw = raw[::-1, :, :]
            labels = labels[::-1, :, :]
        if rng.random() < 0.5:
            raw = raw[:, ::-1, :]
            labels = labels[:, ::-1, :]
        if rng.random() < 0.5:
            raw = raw[:, :, ::-1]
            labels = labels[:, :, ::-1]

        rotations = int(rng.integers(0, 4))
        if rotations:
            raw = np.rot90(raw, rotations, axes=(1, 2))
            labels = np.rot90(labels, rotations, axes=(1, 2))

        return raw, labels

    def _deform_slice(self, image: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        try:
            from scipy.ndimage import gaussian_filter, map_coordinates
        except ImportError:
            return image

        y_coords, x_coords = np.meshgrid(
            np.arange(image.shape[0], dtype=np.float32),
            np.arange(image.shape[1], dtype=np.float32),
            indexing="ij",
        )
        sigma = float(rng.uniform(3.0, 6.0))
        strength = float(rng.uniform(3.0, 8.0))
        flow_y = gaussian_filter(rng.uniform(-1.0, 1.0, image.shape), sigma=sigma) * strength
        flow_x = gaussian_filter(rng.uniform(-1.0, 1.0, image.shape), sigma=sigma) * strength
        return map_coordinates(image, (y_coords + flow_y, x_coords + flow_x), order=1, mode="reflect")

    def _augment_raw(self, raw: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        raw = raw.astype(np.float32, copy=False)
        if raw.max(initial=0.0) > 1.0:
            raw = raw / 255.0

        for z_index in range(raw.shape[0]):
            r = rng.random()
            if r < self.p_drop_slice:
                raw[z_index] = 0.0
            elif r < self.p_drop_slice + self.p_low_contrast:
                mean = float(raw[z_index].mean())
                raw[z_index] = (raw[z_index] - mean) * 0.1 + mean
            elif r < self.p_drop_slice + self.p_low_contrast + self.p_deform_slice:
                raw[z_index] = self._deform_slice(raw[z_index], rng)

        if rng.random() < 0.8:
            raw = raw * float(rng.uniform(0.75, 1.25)) + float(rng.uniform(-0.1, 0.1))
        if rng.random() < 0.25:
            gamma = float(rng.uniform(0.75, 1.5))
            raw = np.power(np.clip(raw, 0.0, 1.0), gamma)
        if rng.random() < 0.15:
            raw = raw + rng.normal(0.0, 0.03, size=raw.shape).astype(np.float32)
        if raw.shape[0] > 2 and rng.random() < 0.05:
            z = int(rng.integers(1, raw.shape[0] - 1))
            raw[z] = 0.5 * (raw[z - 1] + raw[z + 1])

        return np.clip(raw, 0.0, 1.0)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        rng = self._rng_for_index(index)
        volume, origin = self._sample_patch_origin(rng)
        z, y, x = origin
        depth, height, width = self.patch_shape
        raw_z, raw_y, raw_x = volume.raw_start

        handle = self._handle(volume.path)
        raw_dataset = handle[volume.raw_key]
        label_dataset = handle[volume.label_key]

        raw = np.asarray(raw_dataset[raw_z + z : raw_z + z + depth, raw_y + y : raw_y + y + height, raw_x + x : raw_x + x + width])
        labels = np.asarray(label_dataset[z : z + depth, y : y + height, x : x + width])

        if self.augment:
            raw, labels = self._augment_spatial(raw, labels, rng)
            raw = self._augment_raw(raw, rng)

        raw = normalize_raw(raw, self.normalize)

        affinities, mask = labels_to_affinities(
            np.ascontiguousarray(labels),
            offsets=self.offsets,
            target_mode=self.target_mode,
        )
        raw = np.ascontiguousarray(raw[None, ...], dtype=np.float32)
        return torch.from_numpy(raw), torch.from_numpy(affinities), torch.from_numpy(mask)
