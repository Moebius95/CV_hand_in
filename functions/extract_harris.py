import numpy as np
from scipy import signal
import cv2
import scipy

# Harris corner detector
def extract_harris(img, sigma=1.0, k=0.05, thresh=1e-5):
    '''
    Inputs:
    - img:      (h, w) gray-scaled image
    - sigma:    smoothing Gaussian sigma. suggested values: 0.5, 1.0, 2.0
    - k:        Harris response function constant. suggest interval: (0.04 - 0.06)
    - thresh:   scalar value to threshold corner strength. suggested interval: (1e-6 - 1e-4)
    Returns:
    - corners:  (q, 2) numpy array storing the keypoint positions [x, y]
    - C:     (h, w) numpy array storing the corner strength
    '''
    # Convert to float
    img = img.astype(float) / 255.0

    # Compute image gradients
    M_y = np.array([[0, 1, 0],
                    [0, 0, 0],
                    [0, -1, 0]])
    M_x = np.array([[0, 0, 0],
                    [-1, 0, 1],
                    [0, 0, 0]])

    N_p = np.array([[1, 1, 1],
                    [1, 0, 1],
                    [1, 1, 1]])
    
    # You may refer to scipy.signal.convolve2d for the convolution.
    # Do not forget to use the mode "same" to keep the image size unchanged.
    I_x = signal.convolve2d(img, M_x, boundary='symm', mode='same')
    I_y = signal.convolve2d(img, M_y, boundary='symm', mode='same')
    I_xx = I_x * I_x
    I_yy = I_y * I_y
    I_xy = I_x * I_y
    
    # Compute local auto-correlation matrix
    # You may refer to cv2.GaussianBlur for the gaussian filtering (border_type=cv2.BORDER_REPLICATE)
    w_p = cv2.GaussianBlur(img, ksize=[3, 3], sigmaX=sigma, borderType=cv2.BORDER_REPLICATE)
    M_11 = signal.convolve2d(w_p * I_xx, N_p, mode='same')
    M_12 = signal.convolve2d(w_p * I_xy, N_p, mode='same')
    M_22 = signal.convolve2d(w_p * I_yy, N_p, mode='same')
    
    # Compute Harris response function
    C_ij = M_11 * M_22 - M_12 * M_12 - k * ((M_11 + M_22) ** 1)
    # Detection with threshold
    # For the local maximum check, you may refer to scipy.ndimage.maximum_filter to check a 3x3 neighborhood.
    C_max = scipy.ndimage.maximum_filter(C_ij, size = 3)
    corners = np.array(list(zip(np.where(C_max > thresh)[1],np.where(C_max > thresh)[0])))
    return corners, C_max
