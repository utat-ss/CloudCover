import numpy as np

from cloud_detection import *
from config import *
from load_datacube import load_datacube

if __name__ == '__main__':
    # Initialization - Load data
    radiance_data, data_dimensions, wavelength, wavelength_increment = load_datacube(DATA_FOLDER + DATACUBE)

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
    cloud_mask = create_cloud_mask(radiance_data, selected_band, selected_threshold)
    cloud_cover_ratio = measure_cloud_cover(cloud_mask)
    if SAVE_DATA:
        np.savez_compressed(f'{OUTPUT_FOLDER}cloud_mask', mask = cloud_mask,
                            band_index = np.array(selected_band),
                            threshold = np.array(selected_threshold)
        )
    print(f'Step 3 done, total cloud cover: {(cloud_cover_ratio * 100):.2f}%')

    # Optional Step - Apply cloud mask to original datacube
    masked_radiance_data = apply_cloud_mask(radiance_data, cloud_mask)
    if SAVE_DATA:
        np.savez_compressed(f'{OUTPUT_FOLDER}masked_datacube', masked_datacube = masked_radiance_data)
    print('Cloud mask applied')