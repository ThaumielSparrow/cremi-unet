from pathlib import Path
from PIL import Image
from torch.utils.data import Dataset
import numpy as np
import matplotlib.pyplot as plt


def image_index(path):
    return int(path.stem.split('_')[-1])


class LoadData(Dataset):
    def __init__(self, em_dir, seg_dir, transform=None, slice_radius=0, context_dirs=None, volume_depth=None):
        self.em_dir = Path(em_dir)
        self.seg_dir = Path(seg_dir)
        self.transform = transform
        self.slice_radius = slice_radius
        self.volume_depth = volume_depth
        self.context_dirs = [Path(path) for path in (context_dirs or [self.em_dir])]
        self.samples = self._load_samples()
        self.context_images = self._load_context_images()
        self.context_indices = sorted(self.context_images)

    def _load_samples(self):
        samples = []
        for em_path in sorted(self.em_dir.glob('EM_*.png'), key=image_index):
            suffix = em_path.stem.removeprefix('EM_')
            seg_path = self.seg_dir / f'SEG_{suffix}.png'
            if not seg_path.exists():
                raise FileNotFoundError(f'Missing segmentation mask for {em_path}: expected {seg_path}')
            samples.append((em_path, seg_path))

        if not samples:
            raise FileNotFoundError(f'No EM_*.png files found in {self.em_dir}')

        return samples

    def _load_context_images(self):
        context_images = {}
        for context_dir in self.context_dirs:
            for em_path in sorted(context_dir.glob('EM_*.png'), key=image_index):
                context_images.setdefault(image_index(em_path), em_path)

        if not context_images:
            raise FileNotFoundError(f'No EM_*.png context files found in {self.context_dirs}')

        return context_images

    def _volume_bounds(self, center_idx):
        if self.volume_depth is None:
            return None

        volume_start = (center_idx // self.volume_depth) * self.volume_depth
        return volume_start, volume_start + self.volume_depth - 1

    def _context_path(self, center_idx, image_idx):
        bounds = self._volume_bounds(center_idx)
        if bounds is not None:
            image_idx = min(max(image_idx, bounds[0]), bounds[1])

        if image_idx in self.context_images:
            return self.context_images[image_idx]

        candidate_indices = self.context_indices
        if bounds is not None:
            candidate_indices = [idx for idx in self.context_indices if bounds[0] <= idx <= bounds[1]]
            if not candidate_indices:
                raise FileNotFoundError(f'No context images found within volume bounds {bounds}')

        nearest_idx = min(candidate_indices, key=lambda idx: abs(idx - image_idx))
        return self.context_images[nearest_idx]

    def _load_context_stack(self, em_path):
        center_idx = image_index(em_path)
        images = []
        for offset in range(-self.slice_radius, self.slice_radius + 1):
            path = self._context_path(center_idx, center_idx + offset)
            images.append(np.array(Image.open(path).convert('L')))

        return np.stack(images, axis=-1)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        em_path, seg_path = self.samples[idx]
        if self.slice_radius > 0:
            image = self._load_context_stack(em_path)
        else:
            image = np.array(Image.open(em_path).convert('RGB'))
        mask = np.array(Image.open(seg_path).convert('L'), dtype=np.float32)
        mask[mask == 255.0] = 1.0

        # Albumentations augmentations
        if self.transform is not None:
            augmentations = self.transform(image=image, mask=mask)
            image = augmentations['image']
            mask = augmentations['mask']

        return image, mask



if __name__ == "__main__":
    dataset = LoadData(em_dir='data/train/EM', seg_dir='data/train/SEG', transform=None)
    im1 = dataset[0][0][:,:,1]
    im2 = dataset[1][0][:,:,1]
    fig, (ax1, ax2) = plt.subplots(1,2)
    ax1.imshow(im1, cmap='gray')
    ax2.imshow(im2, cmap='gray')
    plt.show()
    
