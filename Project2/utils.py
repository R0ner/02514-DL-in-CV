import numpy as np

def convert(o):
    if isinstance(o, np.float32):
        return float(o)  
    raise TypeError