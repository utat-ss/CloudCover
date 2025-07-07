import numpy as np

import config

def load_datacube(datacube_filepath, wavelength_filepath = None):
    """
    Load a hyperspectral datacube and its associated wavelengths.

    This function loads a 3D hyperspectral datacube for a given file, along with
    its wavelength information. If no wavelength file is provided, it generates
    a linear distribution of wavelengths between a specified minimum and maximum
    wavelength based on configuration settings.

    Parameters
    ----------
    datacube_filepath : str
        Path to the hyperspectral datacube file (.npy or .npz format)

    wavelength_filepath : str, optional
        Path to a text file containing the wavelength centres. If not provided,
        wavelengths are generated linearly.


    Returns
    -------
    tuple
        - data (ndarray): The hyperspectra datacube (3D numpy array w/ dimensions
          rows, columns, bands)
        - data_dimensions (tuple): Dimensions of the data cube (rows, columns,
          bands)
        - wavelength (ndarray): Array of wavelengths corresponding to the centre
          of each spectral band.
        - wavelength_increment (float): The difference between consecutive
          wavelengths (in nm).
    """
    data = np.load(datacube_filepath)
    data_dimensions = data.shape

    # If no wavelength file is provided, generate wavelengths linearly
    if wavelength_filepath is None:
        wavelength = np.linspace(config.MIN_WAVELENGTH, config.MAX_WAVELENGTH, config.NUM_BANDS)
    else:
        wavelength = np.loadtxt(wavelength_filepath)

    wavelength_increment =  wavelength[1] - wavelength[0]

    return data, data_dimensions, wavelength, wavelength_increment

def apply_cloud_mask(radiance_data: np.ndarray, cloud_mask: np.ndarray) -> np.ndarray:
    """
    Applies a binary cloud mask to a hyperspectral datacube.

    All pixels marked as cloud (mask == 1) are set to 0 across all spectral bands.

    Parameters
    ----------
    radiance_data : np.ndarray
        A hyperspectral datacube (3D numpy array w/ dimensions rows, columns,
        bands).
    cloud_mask : np.ndarray
        A 2D binary mask (rows x cols) where cloud pixels are marked as 1.

    Returns
    -------
    np.ndarray
        A masked datacube of the same shape, with cloud pixels zeroed out.
    """
    masked_data = radiance_data.copy()
    masked_data[cloud_mask == 1] = 0

    return masked_data

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
