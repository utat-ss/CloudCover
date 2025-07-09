from math import ceil

from matplotlib import pyplot as plt
from matplotlib.widgets import Button, Slider, TextBox
import numpy as np

from config import *
from utility import load_datacube, apply_cloud_mask_to_band
import cloud_detection

def visualize_band(band: int, datacube: np.ndarray = None):
    """
    Display a single spectral band from the datacube. The band is displayed as
    a grayscale image.
    
    Parameters
    ----------
    band : int
        The index of the band to visualize (0-indexed).
    datacube : np.ndarray
        The datacube to display, if one is not passed in, the configured datacube
        will be loaded.
    """
    if datacube is None:
        data, _, _, _ = load_datacube(DATA_FOLDER + DATACUBE)
    else:
        data = datacube

    data_slice = data[:, :, band]
    max_value = np.max(data_slice)

    plt.imshow(data_slice, cmap='gray', vmin=0, vmax=max_value)
    plt.title(f'Band {band + 1}')

    plt.show()

def visualize_cloud_mask(cloud_mask: np.ndarray = None):
    """
    Displays the binary cloud mask as a grayscale image.
    
    Parameters
    ----------
    cloud_mask : np.ndarray
        The binary cloud mask to display, if one is not passed in, the final cloud
        mask in the output folder will be used.
    """
    if cloud_mask is None:
        cloud_mask = np.load(f'{OUTPUT_FOLDER}final_cloud_mask.npz')['mask']

    plt.imshow(cloud_mask, cmap='gray')
    plt.title('Cloud Mask')

    plt.show()

def visualize_masked_band(band: int, masked_datacube: np.ndarray = None):
    """
    Displays a single spectral band from the masked datacube. The band is
    displayed as a grayscale image.
    
    Parameters
    ----------
    band : int
        The index of the band to visualize (0-indexed).
    masked_datacube : np.ndarray
        The masked datacube to display, if one is not passed in, the masked
        datacube in the output folder will be used.
    """
    if masked_datacube is None:
        masked_data = np.load(f'{OUTPUT_FOLDER}masked_datacube.npz')['masked_datacube']
    else:
        masked_data = masked_datacube

    data_slice = masked_data[:, :, band]
    max_value = np.max(data_slice)

    plt.imshow(data_slice, cmap='gray', vmin=0, vmax=max_value)
    plt.title(f'Band {band + 1}')

    plt.show()

def visualize_datacube_comparison(
    datacube: np.ndarray = None, masked_datacube: np.ndarray = None,
    cloud_mask: np.ndarray = None, band_index: int = None, threshold: float = None
):
    """
    Displays plots of the original datacube, cloud mask, and masked datacube.

    Image plots of the original and masked datacube for a given band, are
    displayed along with the cloud mask. The user can use the slider to switch
    between spectral bands to display.

    Parameters
    ----------
    datacube : np.ndarray
        The datacube to display, if one is not passed in, the configured datacube
        will be loaded.
    masked_datacube : np.ndarray
        The masked datacube to display, if one is not passed in, the masked
        datacube in the output folder will be used.
    cloud_mask : np.ndarray
        The binary cloud mask to display, if one is not passed in, the final cloud
        mask in the output folder will be used.
    band_index : int
        The band index that the cloud mask is based off of (0-indexed), should be
        specified if a cloud mask is passed in.
    threshold : float
        The threshold used to create the cloud mask, should be specified if a cloud
        mask is passed in.
    """
    # ========== Load Data to Display ==========
    if datacube is None:
        radiance_data, _, _, _ = load_datacube(DATA_FOLDER + DATACUBE)
    else:
        radiance_data = datacube

    if masked_datacube is None:
        masked_radiance_data = np.load(f'{OUTPUT_FOLDER}masked_datacube.npz')['masked_datacube']
    else:
        masked_radiance_data = masked_datacube

    if cloud_mask is None:
        cloud_mask = np.load(f'{OUTPUT_FOLDER}final_cloud_mask.npz')['mask']
        cloud_core_mask_data = np.load(f'{OUTPUT_FOLDER}cloud_core_mask.npz')
        band_index = cloud_core_mask_data['band_index']
        threshold = cloud_core_mask_data['threshold']
    elif cloud_mask is not None and (band_index is None or threshold is None):
        print('The band index and threshold used for masking should be specified')
        band_index = -1
        threshold = -1

    # ========== Set Up Figure and Axes ==========
    fig, ax = plt.subplots(ncols=3, sharex=True, sharey=True)
    displayed_band = 0

    # Display band
    data_slice = radiance_data[:, :, displayed_band]
    original_im = ax[0].imshow(data_slice, cmap='gray', vmin=0, vmax=np.max(data_slice))
    ax[0].set_title(f'Original Data, Band: {displayed_band + 1}')

    # Display mask
    ax[1].imshow(cloud_mask, cmap='gray')
    ax[1].set_title(f'Cloud Mask')
    ax[1].text(
        0.5, -0.1,  # X, Y in axes coordinates (0 to 1)
        f'Mask created using Band: {band_index + 1}, Threshold: {threshold:.2f}',
        transform=ax[1].transAxes,
        ha='center', va='top',
        fontsize=10, color='gray'
    )

    # Display masked band
    masked_data_slice = masked_radiance_data[:, :, displayed_band]
    masked_im = ax[2].imshow(masked_data_slice, cmap='gray', vmin=0, vmax=np.max(masked_data_slice))
    ax[2].set_title(f'Masked Data, Band: {displayed_band + 1}')

    # ========== Set up Slider Element ==========
    ax_band_slider = plt.axes([0.2, 0.1, 0.65, 0.03])
    band_slider = Slider(ax_band_slider, 'Band Num', 1, NUM_BANDS, valinit=displayed_band + 1, valstep=1)

    # ========== Slider Callback Function ==========
    def update(val):
        """Updates images and index when the slider is moved."""
        nonlocal displayed_band
        displayed_band = int(band_slider.val) - 1

        # Update original band
        new_data_slice = radiance_data[:, :, displayed_band]
        original_im.set_data(new_data_slice)
        original_im.set_clim(vmin=0, vmax=np.max(new_data_slice))
        ax[0].set_title(f'Original Data, Band: {displayed_band + 1}')

        # Update masked band
        new_masked_data_slice = masked_radiance_data[:, :, displayed_band]
        masked_im.set_data(new_masked_data_slice)
        masked_im.set_clim(vmin=0, vmax=np.max(new_masked_data_slice))
        ax[2].set_title(f'Masked Data, Band: {displayed_band + 1}')

        fig.canvas.draw_idle()

    # ========== Register Slider Event ==========
    band_slider.on_changed(update)

    plt.subplots_adjust(left=0.1, right=0.9, bottom=0.25, wspace=0.5)
    plt.show()

def visualize_interactive_thresholding(radiance_data: np.ndarray) -> tuple[int, float]:
    """
    Interactively visualizes the cloud cover detection process using thresholding.

    This function allows users to visualize the radiance data, cloud mask, and
    the resulting masked data. Users can select the parameters, such as the band
    number and threshold value, to base the cloud mask off of. The user can also
    use the sliders to switch between spectral bands to display and experiment
    with different parameters to base the mask off of.

    Parameters
    ----------
    radiance_data : np.ndarray
        A hyperspectral datacube (3D numpy array w/ dimensions rows, columns,
        bands).

    Returns
    -------
    tuple[int, float]
        - band_index (int): The band used to create the cloud mask.
        - threshold (float): The threshold used to create the cloud mask.
    """
    displayed_band = 0
    mask_band = 0
    mask_threshold = 0

    cloud_mask = cloud_detection.create_cloud_mask(radiance_data, mask_band, mask_threshold)

    # ========== Set Up Figure and Axes ==========
    fig, ax = plt.subplots(ncols=3, sharex=True, sharey=True)
    fig.text(0.5, 0.96,
             'Adjust the mask properties by either using the mask sliders or the textboxes and the enter button. ' \
             'View different bands of the datacube by using the display band slider.\nClose the window to save the parameters.',
             ha='center', va='top', color='gray', fontsize=10)

    # Display band from original data
    data_slice = radiance_data[:, :, displayed_band]
    original_im = ax[0].imshow(data_slice, cmap='gray', vmin=0, vmax=np.max(data_slice))
    ax[0].set_title(f'Original Data, Band: {displayed_band + 1}')

    # Display cloud mask
    cloud_mask_im = ax[1].imshow(cloud_mask, cmap='gray')
    ax[1].set_title(f'Cloud Mask (Band: {mask_band + 1}, Threshold: {mask_threshold:.2f})')

    # Display band from masked data
    masked_data_slice = apply_cloud_mask_to_band(radiance_data, displayed_band, cloud_mask)
    masked_im = ax[2].imshow(masked_data_slice, cmap='gray', vmin=0, vmax=np.max(masked_data_slice))
    ax[2].set_title(f'Masked Data, Band: {displayed_band + 1}')

    # ========== Set Up UI Elements ==========
    # -- Textboxes --
    ax_band_textbox = plt.axes([0.25, 0.25, 0.08, 0.04])
    band_textbox = TextBox(ax_band_textbox, 'Mask Band ', initial=str(mask_band + 1))

    ax_threshold_textbox = plt.axes([0.47, 0.25, 0.1, 0.04])
    threshold_textbox = TextBox(ax_threshold_textbox, 'Mask Threshold ', initial=str(mask_threshold))

    # -- Button --
    ax_enter_button = plt.axes([0.65, 0.25, 0.1, 0.04])
    enter_button = Button(ax_enter_button, 'Enter')

    # -- Sliders --
    ax_mask_band_slider = plt.axes([0.2, 0.15, 0.6, 0.03])
    mask_band_slider = Slider(ax_mask_band_slider, 'Mask Band ', 1, NUM_BANDS, valinit=mask_band + 1, valstep=1)

    ax_mask_threshold_slider = plt.axes([0.2, 0.1, 0.6, 0.03])
    mask_threshold_slider = Slider(ax_mask_threshold_slider, 'Mask Threshold', 0, ceil(np.max(radiance_data)), valinit=mask_threshold, valstep=0.5)

    ax_display_band_slider = plt.axes([0.2, 0.05, 0.6, 0.03])
    display_band_slider = Slider(ax_display_band_slider, 'Displayed Band ', 1, NUM_BANDS, valinit=displayed_band + 1, valstep=1)

    # ========== Helper Function for Callbacks ==========
    def update_cloud_mask():
        """Updates the cloud mask and all plots based on the band and threshold values."""
        nonlocal cloud_mask

        cloud_mask = cloud_detection.create_cloud_mask(radiance_data, mask_band, mask_threshold)

        cloud_mask_im.set_data(cloud_mask)
        cloud_mask_im.set_clim(vmin=0, vmax=1)
        ax[1].set_title(f'Cloud Mask (Band: {mask_band + 1}, Threshold: {mask_threshold:.2f})')

        masked_data_slice = apply_cloud_mask_to_band(radiance_data, displayed_band, cloud_mask)
        masked_im.set_data(masked_data_slice)
        masked_im.set_clim(vmin=0, vmax=np.max(masked_data_slice))
        ax[2].set_title(f'Masked Data, Band: {displayed_band + 1}')

        fig.canvas.draw_idle()

    # ========== Button Callback Function ==========
    def on_enter_button_click(event):
        """Updates the cloud mask based on the band and threshold specified."""
        nonlocal mask_band, mask_threshold

        mask_band = int(band_textbox.text) - 1
        mask_band_slider.set_val(mask_band + 1)

        mask_threshold = float(threshold_textbox.text)
        mask_threshold_slider.set_val(mask_threshold)

        update_cloud_mask()        

    # ========== Slider and UI Callback Functions ==========
    def update_mask_band_slider(val):
        """Updates the cloud mask's band when the mask band slider is moved."""
        nonlocal mask_band

        mask_band = int(mask_band_slider.val) - 1
        band_textbox.set_val(mask_band + 1)

        update_cloud_mask()
        
    def update_mask_threshold_slider(val):
        """Updates the cloud mask's threshold when the mask threshold slider is moved."""
        nonlocal mask_threshold

        mask_threshold = mask_threshold_slider.val
        threshold_textbox.set_val(mask_threshold)

        update_cloud_mask()

    def update_display_band_slider(val):
        """Updates images and index when the display band slider is moved."""
        nonlocal displayed_band
        displayed_band = int(display_band_slider.val) - 1

        # Update original band
        new_data_slice = radiance_data[:, :, displayed_band]
        original_im.set_data(new_data_slice)
        original_im.set_clim(vmin=0, vmax=np.max(new_data_slice))
        ax[0].set_title(f'Original Data, Band: {displayed_band + 1}')

        # Update masked band
        new_masked_data_slice = apply_cloud_mask_to_band(radiance_data, displayed_band, cloud_mask)
        masked_im.set_data(new_masked_data_slice)
        masked_im.set_clim(vmin=0, vmax=np.max(new_masked_data_slice))
        ax[2].set_title(f'Masked Data, Band: {displayed_band + 1}')

        fig.canvas.draw_idle()

    # ========== Register Callbacks and Events ==========
    mask_band_slider.on_changed(update_mask_band_slider)
    mask_threshold_slider.on_changed(update_mask_threshold_slider)
    display_band_slider.on_changed(update_display_band_slider)
    enter_button.on_clicked(on_enter_button_click)

    plt.subplots_adjust(left=0.1, right=0.9, bottom=0.4, wspace=0.5)
    plt.show()

    return mask_band, mask_threshold

if __name__ == '__main__':
    visualize_datacube_comparison()