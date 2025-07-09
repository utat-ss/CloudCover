from math import ceil

from matplotlib import pyplot as plt
from matplotlib.patches import Circle
from matplotlib.widgets import Button, Slider, TextBox
import numpy as np
from scipy.ndimage import binary_closing, binary_dilation
from skimage.feature import graycomatrix, graycoprops
from skimage.filters import threshold_otsu
from skimage.measure import label, regionprops

import config
import utility

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

def perform_interactive_thresholding(radiance_data: np.ndarray) -> tuple[int, float]:
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

    cloud_mask = create_cloud_mask(radiance_data, mask_band, mask_threshold)

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
    masked_data_slice = utility.apply_cloud_mask_to_band(radiance_data, displayed_band, cloud_mask)
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
    mask_band_slider = Slider(ax_mask_band_slider, 'Mask Band ', 1, config.NUM_BANDS, valinit=mask_band + 1, valstep=1)

    ax_mask_threshold_slider = plt.axes([0.2, 0.1, 0.6, 0.03])
    mask_threshold_slider = Slider(ax_mask_threshold_slider, 'Mask Threshold', 0, ceil(np.max(radiance_data)), valinit=mask_threshold, valstep=0.5)

    ax_display_band_slider = plt.axes([0.2, 0.05, 0.6, 0.03])
    display_band_slider = Slider(ax_display_band_slider, 'Displayed Band ', 1, config.NUM_BANDS, valinit=displayed_band + 1, valstep=1)

    # ========== Helper Function for Callbacks ==========
    def update_cloud_mask():
        """Updates the cloud mask and all plots based on the band and threshold values."""
        nonlocal cloud_mask

        cloud_mask = create_cloud_mask(radiance_data, mask_band, mask_threshold)

        cloud_mask_im.set_data(cloud_mask)
        cloud_mask_im.set_clim(vmin=0, vmax=1)
        ax[1].set_title(f'Cloud Mask (Band: {mask_band + 1}, Threshold: {mask_threshold:.2f})')

        masked_data_slice = utility.apply_cloud_mask_to_band(radiance_data, displayed_band, cloud_mask)
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
        new_masked_data_slice = utility.apply_cloud_mask_to_band(radiance_data, displayed_band, cloud_mask)
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

def generate_texture_image(
    radiance_data: np.ndarray, band: int, window_size: int,
    levels: int, offset: int, angle: float
) -> np.ndarray:
    """
    Generates a texture image based on 2nd-order varince using the Grey-Level
    Co-Occurance Matrix (GLCM) for a given hyperspectral band.

    This function computes the local texture variance by sliding a window across
    the image, calculating the GLCM within each window, and calculating the
    variance to highlight texture patterns, such as cloud edges.

    Parameters
    ----------
    radiance_data : np.ndarray
        A hyperspectral datacube (3D numpy array w/ dimensions rows, columns,
        bands).
    band : int
        The index of the spectral band to base the texture image off of.
    window_size : int
        Size of the square window.
    levels : int
        Number of gray levels to convert the band data into before GLCM calculation.
    offset : int
        Pixel distance to consider when computing GLCM.
    angle : float
        Angle in radians that defines the direction of neighbour comparison.

    Returns
    -------
    np.ndarray
        A 2D array (same spatial dimensions) where each pixel contains the local
        GLCM-based variance value computed from its window.
    """
    num_rows, num_cols, _, = radiance_data.shape
    data_slice = radiance_data[:, :, band]
    texture_image = np.zeros((num_rows, num_cols))

    # Convert image to integer values since computing the GLCM requires discrete input
    data_slice_norm = (data_slice - data_slice.min()) / (data_slice.max() - data_slice.min()) # [0, 1] 
    data_slice_int = (data_slice_norm * (levels - 1)).astype(np.uint8)

    # Pad image so that a window can be centred at every pixel in original image
    pad_width = window_size // 2
    data_slice_int_padded = np.pad(data_slice_int, pad_width, mode='reflect')

    # Iterate only through pixels in original image
    for row in range(pad_width, num_rows + pad_width):
        row_start = row - pad_width
        row_end = row + pad_width + 1

        for col in range(pad_width, num_cols + pad_width):
            col_start = col - pad_width
            col_end = col + pad_width + 1
            window = data_slice_int_padded[row_start:row_end, col_start:col_end]

            glcm = graycomatrix(
                window,
                distances=[offset],
                angles=[angle],
                levels=levels,
                symmetric=True,
                normed=True
            )

            texture_image[row - pad_width, col - pad_width] = graycoprops(glcm, prop='variance')[0, 0]

    return texture_image

def create_cloud_margin_mask(
    texture_image: np.ndarray, initial_cloud_mask: np.ndarray,
    min_area: int = 5, dilation_iteration: int = 2
) -> np.ndarray:
    """
    Creates a binary mask of cloud margins using local texture variations and
    an existing cloud mask.

    This function uses rule-based object classification to detect cloud margins
    from a texture image (e.g. derived from GLCM variance). Pixels with value
    1 are identified as cloud margins, and 0 as clear areas.

    Parameters
    ----------
    texture_image : np.ndarray
        A 2D texture image derived from GLCM variance.
    initial_cloud_mask : np.ndarray
        An initial cloud mask created from thresholding radiance.
    min_area : int
        Minimum area (in pixels) for an object to be retained in the margin mask.
    dilation_iteration : int
        Number of iterations to dilate the initial cloud mask.

    Returns
    -------
    np.ndarray
        A 2D binary cloud margin mask with the same dimensions as the texture image.
    """
    margin_mask = np.zeros(texture_image.shape, dtype=np.uint8)

    # Threshold texture image to identify areas of high local variation
    threshold_value = threshold_otsu(texture_image)
    margin_candidates = texture_image > threshold_value

    # Label connected regions
    labeled_regions = label(margin_candidates)
    regions = regionprops(labeled_regions)

    # Filter regions by area
    area_filtered_mask = np.zeros_like(margin_mask)
    for region in regions:
        if region.area >= min_area:
            area_filtered_mask[labeled_regions == region.label] = 1

    # Filter regions by spatial proximity
    dilated_cloud_mask = binary_dilation(initial_cloud_mask, iterations=dilation_iteration)
    proximity_filtered_mask = area_filtered_mask & dilated_cloud_mask

    # Morphological cleaning
    margin_mask = binary_closing(proximity_filtered_mask)

    return margin_mask

def perform_manual_refinement(radiance_data: np.ndarray, cloud_mask: np.ndarray) -> np.ndarray:
    """
    Launches an interactive tool for manually refining a cloud mask.

    This function allows the user to inspect and edit a binary cloud mask by
    painting directly on the mask. Cloud pixels can be added or removed using
    a brush tool directly on the masked image.

    Parameters
    ----------
    radiance_data : np.ndarray
        A hyperspectral datacube (3D numpy array w/ dimensions (rows, columns,
        bands)).
    cloud_mask : np.ndarray
        A 2D binary mask (rows, cols) where cloud pixels are marked as 1.

    Returns
    -------
    np.ndarray
        A 2D binary array representing the refined cloud mask after manual editing.
    """
    is_drawing = False
    paint_mode = False
    paint_value = 1
    brush_size = 1

    band_index = 0
    edited_mask = cloud_mask.copy()
    height, width = radiance_data[:, :, band_index].shape

    # ========== Set Up Figure and Axes ==========
    fig, ax = plt.subplots(ncols=3, sharex=True, sharey=True)
    fig.text(
        0.5, 0.95,
        'Enable painting to add/remove clouds on the masked (right) image. Adjust brush and band settings below.',
        ha='center', va='top', color='gray', fontsize=10
    )

    original_im = ax[0].imshow(radiance_data[:, :, band_index], cmap='gray')
    ax[0].set_title(f'Original Datacube, Band: {band_index + 1}')

    mask_im = ax[1].imshow(edited_mask, cmap='gray')
    ax[1].set_title(f'Cloud Mask')

    masked_band = utility.apply_cloud_mask_to_band(radiance_data, band_index, edited_mask)
    masked_im = ax[2].imshow(masked_band, cmap='gray')
    ax[2].set_title(f'Masked Datacube, Band: {band_index + 1}')

    # ========== Set Up UI Elements ==========
    # -- Sliders --
    ax_band_slider = plt.axes([0.2, 0.28, 0.6, 0.03])
    band_slider = Slider(ax_band_slider, 'Displayed Band ', 1, config.NUM_BANDS, valinit=band_index + 1, valstep=1)

    ax_brush_size_slider = plt.axes([0.3, 0.12, 0.4, 0.03])
    brush_size_slider = Slider(ax_brush_size_slider, 'Brush Size ', 1, 10, valinit=1, valstep=1)

    # -- Buttons --
    ax_toggle_paint_button = plt.axes([0.125, 0.2, 0.2, 0.03])
    toggle_paint_button = Button(ax_toggle_paint_button, 'Turn Painting On')

    ax_add_button = plt.axes([0.375, 0.2, 0.2, 0.03])
    add_button = Button(ax_add_button, 'Add Cloud')

    ax_erase_button = plt.axes([0.625, 0.2, 0.2, 0.03])
    erase_button = Button(ax_erase_button, 'Erase Cloud')

    # ========== Set Up Brush Preview ==========
    brush_preview = Circle((0, 0), radius=brush_size, edgecolor='red', facecolor='none', linewidth=1.5, linestyle='--')
    ax[2].add_patch(brush_preview)
    brush_preview.set_visible(False)

    # ========== Slider and UI Callback Functions ==========
    def update_band_slider(val):
        """Updates images and index when band slider is moved."""
        nonlocal band_index
        band_index = int(band_slider.val) - 1

        # Update original band
        new_data_slice = radiance_data[:, :, band_index]
        original_im.set_data(new_data_slice)
        original_im.set_clim(vmin=0, vmax=np.max(new_data_slice))
        ax[0].set_title(f'Original Datacube, Band: {band_index + 1}')

        # Update masked band
        new_masked_data_slice = utility.apply_cloud_mask_to_band(radiance_data, band_index, edited_mask)
        masked_im.set_data(new_masked_data_slice)
        masked_im.set_clim(vmin=0, vmax=np.max(new_masked_data_slice))
        ax[2].set_title(f'Masked Datacube, Band: {band_index + 1}')

        fig.canvas.draw_idle()

    def update_brush_size_slider(val):
        """Updates brush size when brush size slider is moved."""
        nonlocal brush_size
        brush_size = int(brush_size_slider.val)
        brush_preview.set_radius(brush_size)

    def update_mask():
        """Updates cloud mask and masked datacube when painting."""
        # Update cloud mask
        mask_im.set_data(edited_mask)

        # Update masked data
        new_masked_data_slice = utility.apply_cloud_mask_to_band(radiance_data, band_index, edited_mask)
        masked_im.set_data(new_masked_data_slice)
        masked_im.set_clim(vmin=0, vmax=np.max(new_masked_data_slice))
        ax[2].set_title(f'Masked Datacube, Band: {band_index + 1}')
        
        fig.canvas.draw_idle()

    # ========== Button Callback Functions ==========
    def on_add_button_click(event):
        """Selects adding clouds to cloud mask."""
        nonlocal paint_value
        paint_value = 1

    def on_erase_button_click(event):
        """Selects erasing clouds to cloud mask."""
        nonlocal paint_value
        paint_value = 0

    def on_toggle_paint_button_click(event):
        """Toggles painting mode."""
        nonlocal paint_mode
        paint_mode = not paint_mode
        toggle_paint_button.label.set_text(f'Turn Painting {'Off' if paint_mode else 'On'}')

    # ========== Painting and Interaction Logic ==========
    def paint(event):
        """Paints a circular set of pixels based on brush size."""
        if not paint_mode:
            return
        
        if event.xdata is None or event.ydata is None:
            return
        
        if not event.inaxes == ax[2]: # Masked datacube axes
            return
        
        x, y = int(event.xdata), int(event.ydata)
        x_min, x_max = max(0, x - brush_size), min(width - 1, x + brush_size)
        y_min, y_max = max(0, y - brush_size), min(height - 1, y + brush_size)

        yy, xx = np.ogrid[y_min:y_max, x_min:x_max]
        distance = np.sqrt((yy - y) ** 2 + (xx - x) ** 2)

        brush = distance <= brush_size
        edited_mask[y_min:y_max, x_min:x_max][brush] = paint_value

        update_mask()

    def on_mouse_press(event):
        """Draws/Erases on a mouse press when painting."""
        nonlocal is_drawing

        if paint_mode and event.button == 1: # left click
            is_drawing = True
            paint(event)
    
    def on_mouse_release(event):
        """Stops drawing/erasing on a mouse release when painting."""
        nonlocal is_drawing
        is_drawing = False

    def on_mouse_motion(event):
        """Draws/Erases when the mouse is pressed + moving, displays a preview when painting."""
        if event.inaxes == ax[2]:
            if event.xdata is None or event.ydata is None:
                return

            if paint_mode:
                brush_preview.center = (event.xdata, event.ydata)
                brush_preview.set_visible(True)

                if is_drawing:
                    paint(event)
                else:
                    fig.canvas.draw_idle()
        else:
            brush_preview.set_visible(False)
            fig.canvas.draw_idle()

    # ========== Register Callbacks and Events ==========
    band_slider.on_changed(update_band_slider)
    brush_size_slider.on_changed(update_brush_size_slider)
    toggle_paint_button.on_clicked(on_toggle_paint_button_click)
    add_button.on_clicked(on_add_button_click)
    erase_button.on_clicked(on_erase_button_click)

    fig.canvas.mpl_connect('button_press_event', on_mouse_press)
    fig.canvas.mpl_connect('button_release_event', on_mouse_release)
    fig.canvas.mpl_connect('motion_notify_event', on_mouse_motion)

    plt.subplots_adjust(left=0.1, right=0.9, bottom=0.4, wspace=0.5)
    plt.show()

    return edited_mask
