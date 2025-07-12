"""
Cloud Detection Algorithm for Hyperspectral Images

This script detects clouds in hyperspectral data by selecting a spectral band
and threshold value to create an initial cloud mask. It then generates a texture
image from the same band and performs rule-based classification with the initial
mask to identify cloud margins. Manual refinement of the combined mask is used
to improve the final result.

Key Steps:
1. Select a spectral band for the core cloud mask
    - Bands with high contrast between clouds and other land cover types work best.

2. Select a threshold value to apply to the selected band
    - Threshold separates cloud pixels from clear-sky pixels.

3. Create the core cloud mask using thresholding
    - Produces a binary mask where pixels above threshold are marked as cloud.

4. Generate a texture image from selected spectral band
    - Pads the input image using reflectance.
    - Uses a sliding window to create a grey-level co-occurrence matrix (GLCM)
      at every pixel in input image.
    - Computes the variance of each GLCM to highlight texture differences.

5. Perform rule-based object classification to detect cloud margins
    - Classifies cloud margins using texture image and initial cloud mask.
    - Thresholds the texture image using Otsu's method to find high-variation areas.
    - Labels connected components in thresholded image.
    - Filters out small regions based on a minimum area.
    - Dilates initial cloud mask and retains only texture regions that overlap with it.
    - Applies morphological closing to smooth and clean final margin mask

6. Manually refine the combined cloud mask
    - Launches an interactive tool enabling user to add or remove cloud pixels.


Optional Steps:
- Apply Mask to Datacube
    - Applies a cloud mask to a datacube by setting all cloud pixels to 0.

- Quantify Cloud Coverage in an Image
    - Calculates the ratio of cloud pixels to total pixels in the image.

Outputs:
    - Binary 2D cloud core mask
    - Binary 2D cloud margin mask
    - Binary 2D final cloud mask
    - Texture image
    - Datacube with final cloud mask applied

Notes:
- Settings (e.g. GLCM parameters, data saving, datacube selection) are managed in `config.py`.
- Steps 1 and 2 can run interactively, controlled by `USE_INTERACTIVE_THRESHOLDING` in config.py
    - Interactive mode selects band and threshold value together (may run slower)
    - If not using interactive mode, default values can be set to skip these steps.

Usage:
- Run `main.py` and follow the on-screen prompts to perform each of the steps outlined above.
"""
import numpy as np

from cloud_detection import *
from config import *
from utility import apply_cloud_mask, load_datacube

if __name__ == '__main__':
    # Initialization - Load data
    radiance_data, data_dimensions, wavelength, wavelength_increment = load_datacube(DATA_FOLDER + DATACUBE)

    if USE_INTERACTIVE_THRESHOLDING:
        # Step 1 + 2, Select a band to base mask off of and threshold for clouds
        selected_band, selected_threshold = perform_interactive_thresholding(radiance_data)
        print(f'Step 1 and 2 done, band selected: {selected_band + 1}, threshold selected: {selected_threshold}')
    else:
        # Step 1 - Select a band to base a mask off of
        selected_band = None # Can manually input a band
        if selected_band is None:
            selected_band = select_spectral_band(radiance_data)
        print(f'Step 1 done, band selected: {selected_band + 1}')

        # Step 2 - Select a threshold for clouds
        selected_threshold = None # Can manually input a threshold
        if selected_threshold is None:
            selected_threshold = select_threshold(radiance_data, selected_band)
        print(f'Step 2 done, threshold selected: {selected_threshold}')

    # Step 3 - Create cloud mask by thresholding selected band
    cloud_core_mask = create_cloud_mask(radiance_data, selected_band, selected_threshold)
    if SAVE_DATA:
        np.savez_compressed(f'{OUTPUT_FOLDER}cloud_core_mask', mask = cloud_core_mask,
                            band_index = np.array(selected_band),
                            threshold = np.array(selected_threshold)
        )
    print('Step 3 done, cloud core mask created')

    # Step 4 - Create texture image (to be used for rule-based object classifcation)
    texture_image = generate_texture_image(radiance_data, selected_band, config.GLCM_WINDOW_SIZE,
                                           config.GLCM_LEVELS, config.GLCM_OFFSET,
                                           config.GLCM_ANGLE)
    if SAVE_DATA:
        np.savez_compressed(f'{OUTPUT_FOLDER}texture_image', texture = texture_image)
    print('Step 4 done, texture image generated')

    # Step 5 - Perform cloud margin classification w/ texture image
    cloud_margin_mask = create_cloud_margin_mask(texture_image, cloud_core_mask)
    if SAVE_DATA:
        np.savez_compressed(f'{OUTPUT_FOLDER}cloud_margin_mask', mask = cloud_margin_mask)
    print('Step 5 done, cloud margin mask created')

    # Step 6 - Manually refine cloud mask
    final_cloud_mask = cloud_core_mask | cloud_margin_mask # Merge cloud and cloud margin masks
    final_cloud_mask = perform_manual_refinement(radiance_data, final_cloud_mask)
    cloud_cover_ratio = measure_cloud_cover(final_cloud_mask)
    if SAVE_DATA:
        np.savez_compressed(f'{OUTPUT_FOLDER}final_cloud_mask', mask = final_cloud_mask)
    print(f'Step 6 done, total cloud cover: {(cloud_cover_ratio * 100):.2f}%')

    # Optional Step - Apply cloud mask to original datacube
    masked_radiance_data = apply_cloud_mask(radiance_data, final_cloud_mask)
    if SAVE_DATA:
        np.savez_compressed(f'{OUTPUT_FOLDER}masked_datacube', masked_datacube = masked_radiance_data)
    print('Cloud mask applied')