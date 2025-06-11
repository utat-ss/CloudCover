from matplotlib import pyplot as plt
from matplotlib.widgets import Button, Slider, TextBox
import numpy as np

import config

def select_spectral_band(radiance_data: np.ndarray) -> int:
    """
    Display an interactive viewer for selecting a spectral band for thresholding.

    A matplotlib window is created, which shows a single band of the datacube.
    It includes a slider widget that allows the user to browse through the
    spectral bands by updating the displayed image. The user can finalize
    their selection by pressing the Enter key or closing the plot window.

    Parameters
    ----------
    radiance_data : np.ndarray
        A hyperspectral datacube (3D numpy array w/ dimensions rows, columns,
        bands).

    Returns
    -------
    int
        The index of the spectral band selected by the user (0-indexed).
    """
    band_index = [0] # Use a list so it can be updated inside nested functions
    data_slice = radiance_data[:, :, band_index[0]]
    max_val = np.max(data_slice)

    fig, ax = plt.subplots()
    plt.subplots_adjust(left=0.2, right=0.8, bottom=0.25)
    im = ax.imshow(data_slice, cmap='gray', vmin=0, vmax=max_val, origin='upper')
    ax.set_title(f'Band: {band_index[0] + 1}')
    fig.text(
        0.5, -0.1,  # X, Y in axes coordinates (0 to 1)
        'Use the slider to browse bands, press \'Enter\' or close the plot to select a band',
        transform=ax.transAxes,
        ha='center', va='top',
        fontsize=10, color='gray'
    )

    ax_band = plt.axes([0.2, 0.1, 0.65, 0.03])
    band_slider = Slider(ax_band, 'Band Num', 1, config.NUM_BANDS, valinit=band_index[0] + 1, valstep = 1)

    # Function to update image and index when slider is moved
    def update(val):
        band_index[0] = int(band_slider.val) - 1
        new_data_slice = radiance_data[:, :, band_index[0]]
        new_max_val = np.max(new_data_slice)

        im.set_data(new_data_slice)
        im.set_clim(vmin=0, vmax=new_max_val)
        ax.set_title(f'Band: {band_index[0] + 1}')
        fig.canvas.draw_idle()

    # Function to use the Enter key to close the plot
    def on_key(event):
        if event.key == 'enter':
            plt.close(fig)

    band_slider.on_changed(update)
    fig.canvas.mpl_connect('key_press_event', on_key)
    plt.show()

    return band_index[0]

def select_threshold(radiance_data: np.ndarray, band: int) -> float:
    """
    Displays a slice of a datacube at the specified spectral band and allows
    the user to click on a pixel to select a threshold value.

    Parameters
    ----------
    radiance_data : np.ndarray
        A hyperspectral datacube (3D numpy array w/ dimensions rows, columns,
        bands).
    band : int
        The index of the spectral band to display.

    Returns
    -------
    float
        The radiance value selected by the user to be used as a threshold.
    """
    data_slice = radiance_data[:, :, band]
    max_value = np.max(data_slice)
    threshold = [0] # Use a list so it can be updated inside nested functions

    fig, ax = plt.subplots()
    plt.subplots_adjust(left=0.2, right=0.8, bottom=0.25)
    im = ax.imshow(data_slice, cmap='gray', vmin=0, vmax=max_value)
    ax.set_title(f'Band: {band + 1}')
    fig.text(
        0.5, -0.1,  # X, Y in axes coordinates (0 to 1)
        'Click on the image to select a threshold, or manually input value',
        transform=ax.transAxes,
        ha='center', va='top',
        fontsize=10, color='gray'
    )

    ax_textbox = plt.axes([0.35, 0.08, 0.15, 0.04])
    textbox = TextBox(ax_textbox, 'Threshold Input:', initial='0')

    ax_button = plt.axes([0.55, 0.08, 0.1, 0.04])
    button = Button(ax_button, 'Enter')

    # Function to register threshold at a mouse click
    def on_mouse_click(event):
        if event.inaxes == ax:
            x, y = int(event.xdata), int(event.ydata)
            threshold[0] = data_slice[y, x]
            plt.close(fig)

    # Function to register threshold from textbox + button
    def on_button_click(event):
        threshold[0] = float(textbox.text)
        plt.close(fig)

    button.on_clicked(on_button_click)
    fig.canvas.mpl_connect('button_press_event', on_mouse_click)
    plt.show()

    return threshold[0]

def create_cloud_mask(radiance_data: np.ndarray, band: int, threshold: float) -> np.ndarray:
    """
    Creates a binary cloud mask based on a selected spectral band and threshold.

    Each pixel's radiance in the specified band is compared against the given
    threshold. If the radiance value is greater than the threshold, the
    corresponding mask pixel is set to 1 (cloud), otherwise, it is set to 0 (clear).

    Parameters
    ----------
    radiance_data : np.ndarray
        A hyperspectral datacube (3D numpy array w/ dimensions rows, columns,
        bands).
    band : int
        The index of the spectral band to use for thresholding.
    threshold: float
        The radiance threshold for cloud detection.
    
        
    Returns
    -------
    np.ndarray
        A 2D binary cloud mask with dimensions: rows x cols.
    """
    num_rows, num_cols, _ = radiance_data.shape
    mask = np.zeros((num_rows, num_cols), dtype=np.uint8)

    for row in range(num_rows):
        for col in range(num_cols):
            mask[row, col] = 1 if radiance_data[row, col, band] > threshold else 0

    return mask

def measure_cloud_cover(cloud_mask: np.ndarray) -> float:
    """
    Calculates the cloud cover in the image as the ratio of cloud pixels to total
    pixels, based on the provided cloud mask.

    Parameters
    ----------
    cloud_mask : np.ndarray
        A binary array where cloud pixels are marked with 1 and non-cloud pixels
        with 0.

    Returns
    -------
    float
        The fraction o fpixels in the image that are classified as clouds.
    """
    num_cloud_pixels = np.sum(cloud_mask)
    num_total_pixels = cloud_mask.size
    cloud_cover_ratio = num_cloud_pixels / num_total_pixels

    return cloud_cover_ratio


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
    num_rows, num_cols, _ = radiance_data.shape
    masked_data = radiance_data.copy()

    for row in range(num_rows):
        for col in range(num_cols):
            if cloud_mask[row, col] == 1:
                masked_data[row, col, :] = 0

    return masked_data