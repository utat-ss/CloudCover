DATA_FOLDER = 'Cloud Detection/data/'
OUTPUT_FOLDER = 'Cloud Detection/data_output/'

SAVE_DATA = False
SAVE_PLOTS = False

USE_INTERACTIVE_THRESHOLDING = True

# Datacube Specifications
DATACUBE = '59-vigo-radiance-small.npy'
NUM_ROWS = 956
NUM_COLS = 684
NUM_BANDS = 120
MIN_WAVELENGTH = 400 # nm
MAX_WAVELENGTH = 800 # nm

# GLCM Parameters
GLCM_WINDOW_SIZE = 5
GLCM_LEVELS = 8
GLCM_OFFSET = 1
GLCM_ANGLE = 0 # radians