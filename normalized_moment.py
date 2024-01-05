from central_moment import central_moment

def normalized_moment(image, i, j):
    """Compute the normalized central moment."""
    
    cm = central_moment(image, i, j)
    
    M00 = central_moment(image, 0, 0)
    
    if M00 == 0:
        return 0.0
    
    N = cm / (M00 ** ((i + j) / 2 + 1))
        
    return N
