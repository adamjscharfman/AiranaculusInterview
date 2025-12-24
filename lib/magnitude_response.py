import numpy as np

def compute_block_magnitude(iq_data:np.array,blocksize:int=1024,hop:int=512,fs:float=1.0)->tuple[np.array,np.array]:
    num_blocks = ((len(iq_data) - blocksize) // hop) + 1
    t = np.arange(0,len(iq_data),1)/fs

    # For loop method (use to validate as_strided approach)
    # mag_block = np.zeros(num_blocks,dtype=iq_data.real.dtype)
    # t_block = np.zeros(num_blocks)

    # for i in range(num_blocks):
    #     idx = i * hop
    #     block = iq_data[idx:idx+blocksize]
    #     mag_block[i] = np.sqrt(np.mean(np.abs(block)**2))
    #     t_block[i] = t[idx + blocksize//2]

    # Use stride tricks to create a view of the data as a matrix with each of the blocks to average over
    shape = (num_blocks,blocksize)
    strides = (iq_data.strides[0]*hop, iq_data.strides[0])
    blocks = np.lib.stride_tricks.as_strided(iq_data, shape=shape, strides=strides)
    mag_block = np.sqrt(np.mean(np.abs(blocks)**2,axis=1))
    t_block = t[blocksize//2::hop][:num_blocks]
    return t_block, mag_block
