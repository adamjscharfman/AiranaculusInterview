import sys,os
sys.path.append('lib')
import parse
import plotters
import numpy as np
import matplotlib.pyplot as plt

filename = 'data/example2_complex_float32.bin'
iq_data = parse.parse_iq_file(filename,np.float32,np.complex64)
# Plot the Signal from the file
# plotters.plot_real_imag(iq_data)

# Plot the FFT of the Signal From File
# f_fft,mag_fft = plotters.plot_fft_magnitude(iq_data)

# Plot the Welch PSD
fs = 1.0 #Not specified, just using a nominal value of 1
nperseg = 512 # Number of samples per frequency response
# nperseg = 1024
# nperseg = 4094
# nperseg = 2048
# noverlap = nperseg//2 #50% Overlap
noverlap = nperseg//4 #25% Overlap
nfft = nperseg # No zero padding on the fft
f_welch,Pxx_welch = plotters.plot_welch_psd(iq_data,nperseg,noverlap,nfft,fs)

#Plot the Spectrogram
plot_time_downsample_factor=4
f_spec,t_spec,Sxx_spec = plotters.plot_spectrogram(iq_data,nperseg,noverlap,nfft,fs,plot_time_downsample_factor)

# Plot the magnitude response averaged over blocks
t_block,mag_block = plotters.plot_block_magnitude(iq_data,nperseg,noverlap,fs)

# Observe signals in the following frequency ranges
#Signal 1: [-0.355,-0.194] * Pi
#Signal 2: [-0.182,-0.10]
#Signal 3: [-0.10,0.065]
#Signal 4: [0.094,0.262]
#Signal 5: [0.296,0.439]
# Can use the Parks Mclellan Algorithm


breakpoint()