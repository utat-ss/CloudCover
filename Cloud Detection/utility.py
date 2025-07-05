import numpy as np

def apply_cloud_mask_to_band(radiance_data: np.ndarray, band: int, cloud_mask: np.ndarray) -> np.ndarray:
    """
    Applies a binary cloud mask to a band of a hyperspectral datacube.

    All pixels marked as cloud (mask == 1) are set to 0 across the image.

    Parameters
    ----------
    radiance_data : np.ndarray
        A hyperspectral datacube (3D numpy array w/ dimensions (rows, columns,
        bands)).
    band : int
        The index of the band to which the mask will be applied.
    cloud_mask : np.ndarray
        A 2D binary mask (rows, cols) where cloud pixels are marked as 1.

    Returns
    -------
    np.ndarray
        A masked version of the specified band from the datacube.
    """
    return np.where(cloud_mask == 1, 0, radiance_data[:, :, band])