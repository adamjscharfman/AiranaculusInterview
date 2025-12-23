import numpy as np

def apply_frequency_shift(iq_data:np.array,fc_offset:float,fs:float):
    #X(w-wfc) <=> x(t)e^(-1j*wfc*t)
    n = np.arange(len(iq_data))
    iq_data_shifted = iq_data * np.exp(-1j*2*np.pi*fc_offset*n/fs)
    return iq_data_shifted

def compute_average_phase(iq_data:np.array,samples_per_symbol:int,offset:int=0):
    num_symbols = int((len(iq_data)-offset)//samples_per_symbol) #Floor
    iq_data_trunc = iq_data[offset:offset+num_symbols*samples_per_symbol] #Make sure data can be reshaped
    iq_data_reshape = np.reshape(iq_data_trunc,(num_symbols,samples_per_symbol))
    # Check
    # np.any(iq_data_reshape[0]  - iq_data_trunc[:samples_per_symbol])  
    phase_mtx = np.angle(iq_data_reshape)
    average_phase = np.mean(phase_mtx,axis=1)
    return average_phase

def estimate_m_psk_rotation(iq_data:np.array,num_bits_per_symbol:int):
    m_psk = 2**num_bits_per_symbol
    rotation = 1/m_psk * np.angle(np.sum(iq_data**m_psk))
    return rotation

def estimate_bpsk_frequency_offset_one_lag_estimator(iq_data:np.array,samples_per_symbol:int):
    z_n = iq_data[1:]*np.conj(iq_data[:-1])
    f_0_est = 1/(2*np.pi*samples_per_symbol) * np.angle(np.sum(z_n))
    return f_0_est
