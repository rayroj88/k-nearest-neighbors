import numpy as np
from raw_moment import raw_moment

def central_moment(image, i, j):
    """Compute the central moment."""
    
    M00 = raw_moment(image, 0, 0)
    
    # Compute the centroid of the image
    if M00 == 0:
        x_bar = 0
        y_bar = 0
    
    else:
        x_bar = raw_moment(image, 1, 0) / M00
        y_bar = raw_moment(image, 0, 1) / M00

    U = np.sum([(x - x_bar)**i * (y - y_bar)**j * image[y, x] for x in range(image.shape[1]) for y in range(image.shape[0])])

    return U
