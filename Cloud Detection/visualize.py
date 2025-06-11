from matplotlib import pyplot as plt
from matplotlib.widgets import Button, Slider, TextBox
import numpy as np

from config import *
from load_datacube import load_datacube
import cloud_detection

def visualize_band(band: int):
    """
    Display a single spectral band from the datacube. The band is displayed as
    a grayscale image.
    
    Parameters
    ----------
    band : int
        The index of the band to visualize (0-indexed).
    """
    radiance_data, _, _, _ = load_datacube(DATA_FOLDER + DATACUBE)

    data_slice = radiance_data[:, :, band]
    max_value = np.max(data_slice)

    plt.imshow(data_slice, cmap='gray', vmin=0, vmax=max_value)
    plt.show()

def visualize_cloud_mask():
    """Displays the binary cloud mask as a grayscale image."""
    cloud_mask = np.load(f'{OUTPUT_FOLDER}cloud_mask.npz')['mask']

    plt.imshow(cloud_mask, cmap='gray')
    plt.show()

def visualize_masked_band(band: int):
    """
    Displays a single spectral band from the masked datacube. The band is
    displayed as a grayscale image.
    
    Parameters
    ----------
    band : int
        The index of the band to visualize (0-indexed).
    """
    masked_radiance_data = np.load(f'{OUTPUT_FOLDER}masked_datacube.npz')['masked_datacube']

    data_slice = masked_radiance_data[:, :, band]
    max_value = np.max(data_slice)

    plt.imshow(data_slice, cmap='gray', vmin=0, vmax=max_value)
    plt.show()

def visualize_datacube_comparison():
    """
    Displays plots of the original datacube, cloud mask, and masked datacube.

    Image plots of the original and masked datacube for a given band, are
    displayed along with the cloud mask. The user can use the slider to switch
    between spectral bands to display.
    """
    # Load data to display
    radiance_data, _, _, _ = load_datacube(DATA_FOLDER + DATACUBE)
    masked_radiance_data = np.load(f'{OUTPUT_FOLDER}masked_datacube.npz')['masked_datacube']
    cloud_mask_data = np.load(f'{OUTPUT_FOLDER}cloud_mask.npz')
    cloud_mask = cloud_mask_data['mask']
    band_index = cloud_mask_data['band_index']
    threshold = cloud_mask_data['threshold']

    fig, ax = plt.subplots(ncols=3)
    displayed_band = [0]

    # Display band
    data_slice = radiance_data[:, :, displayed_band[0]]
    max_value = np.max(data_slice)
    original_im = ax[0].imshow(data_slice, cmap='gray', vmin=0, vmax=max_value)
    ax[0].set_title(f'Original Data, Band: {displayed_band[0] + 1}')

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
    masked_data_slice = masked_radiance_data[:, :, displayed_band[0]]
    masked_max_value = np.max(masked_data_slice)
    masked_im = ax[2].imshow(masked_data_slice, cmap='gray', vmin=0, vmax=masked_max_value)
    ax[2].set_title(f'Masked Data, Band: {displayed_band[0] + 1}')

    ax_band_slider = plt.axes([0.2, 0.1, 0.65, 0.03])
    band_slider = Slider(ax_band_slider, 'Band Num', 1, NUM_BANDS, valinit=displayed_band[0] + 1, valstep=1)

    # Function to update image and index when slider is moved
    def update(val):
        displayed_band[0] = int(band_slider.val) - 1

        # Update original band
        new_data_slice = radiance_data[:, :, displayed_band[0]]
        new_max_val = np.max(new_data_slice)
        original_im.set_data(new_data_slice)
        original_im.set_clim(vmin=0, vmax=new_max_val)
        ax[0].set_title(f'Original Data, Band: {displayed_band[0] + 1}')

        # Update masked band
        new_masked_data_slice = masked_radiance_data[:, :, displayed_band[0]]
        new_masked_max_val = np.max(new_masked_data_slice)
        masked_im.set_data(new_masked_data_slice)
        masked_im.set_clim(vmin=0, vmax=new_masked_max_val)
        ax[2].set_title(f'Masked Data, Band: {displayed_band[0] + 1}')

        fig.canvas.draw_idle()

    band_slider.on_changed(update)
    plt.subplots_adjust(left=0.1, right=0.9, bottom=0.25, wspace=0.5)
    plt.show()

def visualize_interactive_thresholding():
    """
    Interactively visualizes the cloud cover detection process using thresholding.

    This function allows users to visualize the radiance data, cloud mask, and
    the resulting masked data. Users can select the parameters, such as the band
    number and threshold value, to base the cloud mask off of. The user can also
    use the slider to switch between spectral bands to display.
    """
    radiance_data, _, _, _ = load_datacube(DATA_FOLDER + DATACUBE)
    displayed_band = [0]
    mask_band = [0]
    mask_threshold = [0]

    cloud_mask = [cloud_detection.create_cloud_mask(radiance_data, mask_band[0], mask_threshold[0])]
    masked_radiance_data = [cloud_detection.apply_cloud_mask(radiance_data, cloud_mask[0])]

    fig, ax = plt.subplots(ncols=3)

    # Display band
    data_slice = radiance_data[:, :, displayed_band[0]]
    max_value = np.max(data_slice)
    original_im = ax[0].imshow(data_slice, cmap='gray', vmin=0, vmax=max_value)
    ax[0].set_title(f'Original Data, Band: {displayed_band[0] + 1}')

    # Display mask
    cloud_mask_im = ax[1].imshow(cloud_mask[0], cmap='gray')
    ax[1].set_title(f'Cloud Mask (Band: {mask_band[0] + 1}, Threshold: {mask_threshold[0]:.2f})')

    # Display masked band
    masked_data_slice = masked_radiance_data[0][:, :, displayed_band[0]]
    masked_max_value = np.max(masked_data_slice)
    masked_im = ax[2].imshow(masked_data_slice, cmap='gray', vmin=0, vmax=masked_max_value)
    ax[2].set_title(f'Masked Data, Band: {displayed_band[0] + 1}')

    ax_band_slider = plt.axes([0.2, 0.05, 0.6, 0.03])
    band_slider = Slider(ax_band_slider, 'Displayed Band ', 1, NUM_BANDS, valinit=displayed_band[0] + 1, valstep=1)

    ax_band_textbox = plt.axes([0.25, 0.15, 0.08, 0.04])
    band_textbox = TextBox(ax_band_textbox, 'Mask Band ', initial=str(mask_band[0] + 1))

    ax_threshold_textbox = plt.axes([0.47, 0.15, 0.1, 0.04])
    threshold_textbox = TextBox(ax_threshold_textbox, 'Mask Threshold ', initial=str(mask_threshold[0]))

    ax_button = plt.axes([0.65, 0.15, 0.1, 0.04])
    button = Button(ax_button, 'Enter')

    def onclick(event):
        mask_band[0] = int(band_textbox.text) - 1
        mask_threshold[0] = float(threshold_textbox.text)

        cloud_mask[0] = cloud_detection.create_cloud_mask(radiance_data, mask_band[0], mask_threshold[0])
        masked_radiance_data[0] = cloud_detection.apply_cloud_mask(radiance_data, cloud_mask[0])

        cloud_mask_im.set_data(cloud_mask[0])
        cloud_mask_im.set_clim(vmin=0, vmax=1)
        ax[1].set_title(f'Cloud Mask (Band: {mask_band[0] + 1}, Threshold: {mask_threshold[0]:.2f})')

        masked_data_slice = masked_radiance_data[0][:, :, displayed_band[0]]
        masked_max_val = np.max(masked_data_slice)
        masked_im.set_data(masked_data_slice)
        masked_im.set_clim(vmin=0, vmax=masked_max_val)
        ax[2].set_title(f'Masked Data, Band: {displayed_band[0] + 1}')

        fig.canvas.draw_idle()

    # Function to update image and index when slider is moved
    def update(val):
        displayed_band[0] = int(band_slider.val) - 1

        # Update original band
        new_data_slice = radiance_data[:, :, displayed_band[0]]
        new_max_val = np.max(new_data_slice)
        original_im.set_data(new_data_slice)
        original_im.set_clim(vmin=0, vmax=new_max_val)
        ax[0].set_title(f'Original Data, Band: {displayed_band[0] + 1}')

        # Update masked band
        new_masked_data_slice = masked_radiance_data[0][:, :, displayed_band[0]]
        new_masked_max_val = np.max(new_masked_data_slice)
        masked_im.set_data(new_masked_data_slice)
        masked_im.set_clim(vmin=0, vmax=new_masked_max_val)
        ax[2].set_title(f'Masked Data, Band: {displayed_band[0] + 1}')

        fig.canvas.draw_idle()

    band_slider.on_changed(update)
    button.on_clicked(onclick)
    plt.subplots_adjust(left=0.1, right=0.9, bottom=0.25, wspace=0.5)
    plt.show()

if __name__ == '__main__':
    visualize_datacube_comparison()