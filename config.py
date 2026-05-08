MODEL_NAME_AFFINITY = "anisotropic_affinity_unet"

RAW_DATASET_KEY = "volumes/raw"
LABEL_DATASET_KEY = "volumes/labels/neuron_ids"

DEFAULT_OFFSETS = (
    (-1, 0, 0),
    (0, -1, 0),
    (0, 0, -1),
    (-2, 0, 0),
    (0, -3, 0),
    (0, 0, -3),
    (-3, 0, 0),
    (0, -9, 0),
    (0, 0, -9),
    (-4, 0, 0),
    (0, -27, 0),
    (0, 0, -27),
)

DEFAULT_SCALE_FACTORS = (
    (1, 3, 3),
    (1, 3, 3),
    (2, 2, 2),
    (2, 2, 2),
    (2, 2, 2),
)

DEFAULT_TARGET_MODE = "disaffinity"
DEFAULT_NORMALIZATION = "standardize"

DEFAULT_PATCH_SHAPE = (32, 360, 360)
DEFAULT_BLOCK_SHAPE = (32, 512, 512)
DEFAULT_HALO = (4, 64, 64)
