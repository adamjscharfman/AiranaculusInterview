import os
import numpy as np

def parse_iq_file(filename:str,dtype_in:np.dtype,dtype_out:np.dtype):
    data = np.fromfile(filename,dtype_in).astype(dtype_out)
    iq_data = data[0::2] + 1j*data[1::2]
    return iq_data