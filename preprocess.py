import math
import random
import sys
from pathlib import Path
from typing import cast

import cv2 as cv
import h5py
import numpy as np
from tqdm import tqdm


class CustomErrorTypes(Exception):
    # Class that lets me tell you that you did something wrong
    pass

class CREMI:
    """
    Class containing methods to manipulate and process CREMI data. Structure is not extendable beyond datasets at https://cremi.org/data/.

    To initialize, pass a list containing filenames referring to .hdf files.
    """
    RAW_DATASET = 'volumes/raw'
    LABEL_DATASET = 'volumes/labels/neuron_ids'

    def __init__(self, samplefolder, savefolder=None, autocon=False):
        self.samplefolder = Path(samplefolder)
        self.savefolder = Path(savefolder) if savefolder else None
        self.autocon = autocon

    def _prepare_output_dir(self, path):
        path.mkdir(exist_ok=True, parents=True)
        existing_files = [file for file in path.glob('*.png') if file.is_file()]
        if not existing_files:
            return

        if not self.autocon:
            yn_input = input(f'{path} already exists. OK to overwrite its PNG contents? [y/n] ').lower()
            if yn_input == 'n':
                sys.exit()

        for file in existing_files:
            file.unlink()

    @staticmethod
    def _available_datasets(dataset):
        names = []

        def collect(name, obj):
            if isinstance(obj, h5py.Dataset):
                names.append(name)

        dataset.visititems(collect)
        return ', '.join(names)

    def _validate_hdf(self, dataset, path):
        if self.RAW_DATASET not in dataset:
            raise CustomErrorTypes(
                f'{path} does not contain {self.RAW_DATASET}. '
                f'Available datasets: {self._available_datasets(dataset)}'
            )

        if self.LABEL_DATASET not in dataset:
            raise CustomErrorTypes(
                f'{path} does not contain {self.LABEL_DATASET}. '
                'CREMI test volumes are raw-only and cannot be used as validation masks. '
                f'Available datasets: {self._available_datasets(dataset)}'
            )

    @staticmethod
    def _aligned_raw_region(raw_dataset, label_dataset):
        label_shape = label_dataset.shape
        raw_shape = raw_dataset.shape

        label_offset = np.asarray(label_dataset.attrs.get('offset', np.zeros(3)), dtype=float)
        raw_offset = np.asarray(raw_dataset.attrs.get('offset', np.zeros(3)), dtype=float)
        raw_resolution = np.asarray(raw_dataset.attrs.get('resolution', np.ones(3)), dtype=float)
        label_resolution = np.asarray(label_dataset.attrs.get('resolution', raw_resolution), dtype=float)

        if not np.allclose(raw_resolution, label_resolution):
            raise CustomErrorTypes(
                f'Raw and label resolutions differ: raw={raw_resolution}, labels={label_resolution}. '
                'This preprocessing path expects matching voxel resolution.'
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

        if not fits_raw:
            if raw_shape[:3] == label_shape[:3]:
                z_start, y_start, x_start = 0, 0, 0
            else:
                raise CustomErrorTypes(
                    f'Cannot align raw volume shape {raw_shape} with label shape {label_shape}. '
                    f'Computed raw start index was {(z_start, y_start, x_start)}.'
                )

        return z_start, slice(y_start, y_start + height), slice(x_start, x_start + width)

    def preprocess(self):
        """
        Preprocesses superset CREMI data. Collates all datasets passed together.

        Null function. Creates 2 folders, one containing raw 2D EM data, and the other with processed segmentation.
        """
        true_iter = 0
        hdf_paths = sorted(self.samplefolder.rglob('*.hdf'))
        if not hdf_paths:
            raise CustomErrorTypes(f'No .hdf files found under {self.samplefolder}')

        for superset_data in hdf_paths:
            with h5py.File(superset_data, 'r') as dataset:
                self._validate_hdf(dataset, superset_data)

        out_em = (self.savefolder / 'EM') if self.savefolder else Path('EM')
        out_seg = (self.savefolder / 'SEG') if self.savefolder else Path('SEG')
        self._prepare_output_dir(out_em)
        self._prepare_output_dir(out_seg)

        for superset_data in hdf_paths:
            with h5py.File(superset_data, 'r') as dataset:
                raw_dataset = cast(h5py.Dataset, dataset[self.RAW_DATASET])
                label_dataset = cast(h5py.Dataset, dataset[self.LABEL_DATASET])
                z, height, width = label_dataset.shape
                raw_z_start, raw_y_slice, raw_x_slice = self._aligned_raw_region(raw_dataset, label_dataset)

                for z_index in (pbar := tqdm(range(z))):
                    pbar.set_description(f'Processing {superset_data.name}')
                    image = np.asarray(raw_dataset[raw_z_start + z_index, raw_y_slice, raw_x_slice])
                    seg = np.asarray(label_dataset[z_index, :, :])

                    em_savepath = out_em / f'EM_{true_iter}.png'
                    cv.imwrite(str(em_savepath), image)

                    cnts, _ = cv.findContours(seg.astype(np.int32, copy=False), cv.RETR_FLOODFILL, cv.CHAIN_APPROX_SIMPLE)
                    output = np.zeros([height, width], np.uint8)
                    cv.drawContours(output, cnts, -1, 255, 3)
                    seg_savepath = out_seg / f'SEG_{true_iter}.png'
                    cv.imwrite(str(seg_savepath), output)
                    true_iter += 1


    def test_train_split(self, train_folder, test_folder, train_volume=0.5):
        """
        Takes half of training dataset and uses it as evaluation metric. Percentage can be customized by passing
        a frequency to represent the percentage of data that stays as training.
        """
        em_train_folder = Path(train_folder) / 'EM'
        seg_train_folder = Path(train_folder) / 'SEG'
        em_test_folder = Path(test_folder) / 'EM'
        seg_test_folder = Path(test_folder) / 'SEG'

        if not em_train_folder.exists() or not seg_train_folder.exists():
            raise CustomErrorTypes('This function requires train_folder to contain EM and SEG folders')

        self._prepare_output_dir(em_test_folder)
        self._prepare_output_dir(seg_test_folder)

        if not 0 < train_volume < 1:
            raise CustomErrorTypes('train_volume must be between 0 and 1')

        pairs = []
        for em_path in sorted(em_train_folder.glob('EM_*.png')):
            suffix = em_path.stem.removeprefix('EM_')
            seg_path = seg_train_folder / f'SEG_{suffix}.png'
            if not seg_path.exists():
                raise CustomErrorTypes(f'Missing segmentation mask for {em_path.name}: expected {seg_path}')
            pairs.append((em_path, seg_path))

        if not pairs:
            raise CustomErrorTypes(f'No EM_*.png files found in {em_train_folder}')

        train_count = math.ceil(len(pairs) * train_volume)
        if len(pairs) > 1:
            train_count = min(max(train_count, 1), len(pairs) - 1)
        imgs_to_move = len(pairs) - train_count

        for em_path, seg_path in random.sample(pairs, imgs_to_move):
            em_path.rename(em_test_folder / em_path.name)
            seg_path.rename(seg_test_folder / seg_path.name)


if __name__ == '__main__':
    container = CREMI(samplefolder='samples/train', savefolder='data/train', autocon=True)
    container.preprocess()
    container.test_train_split(train_volume=0.75, test_folder='data/test', train_folder='data/train')
