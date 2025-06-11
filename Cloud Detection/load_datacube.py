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