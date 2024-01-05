import numpy as np
from normalized_moment import normalized_moment

def hu_moment(image, m):
    """Compute the m-th Hu moment."""
    
    # Define the normalized moments
    n20 = normalized_moment(image, 2, 0)
    n02 = normalized_moment(image, 0, 2)
    n11 = normalized_moment(image, 1, 1)
    n30 = normalized_moment(image, 3, 0)
    n12 = normalized_moment(image, 1, 2)
    n21 = normalized_moment(image, 2, 1)
    n03 = normalized_moment(image, 0, 3)
    
    # Hu moments
    if m == 1:
        I = n20 + n02
    elif m == 2:
        I = (n20 - n02)**2 + 4*n11**2
    elif m == 3:
        I = (n30 - 3*n12)**2 + (3*n21 - n03)**2
    elif m == 4:
        I = (n30 + n12)**2 + (n03 + n21)**2
    elif m == 5:
        I = (n30 - 3*n12) * (n30 + n12) * ((n30 + n12)**2 - 3*(n21 + n03)**2) + \
            (3*n21 - n03) * (n21 + n03) * (3*(n30 + n12)**2 - (n21 + n03)**2)
    elif m == 6:
        I = (n20 - n02) * ((n30 + n12)**2 - (n21 + n03)**2) + \
            4*n11 * (n30 + n12) * (n21 + n03)
    elif m == 7:
        I = (3*n21 - n03) * (n30 + n12) * ((n30 + n12)**2 - 3*(n21 + n03)**2) - \
            (n30 - 3*n12) * (n21 + n03) * (3*(n30 + n12)**2 - (n21 + n03)**2)
    else:
        I = 0.0
        
    return I