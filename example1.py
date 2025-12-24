import sys,os
sys.path.append('lib')
import parse
import plotters
import demod
import numpy as np
import scipy
import matplotlib.pyplot as plt
from fractions import Fraction

filename = 'data/example1_complex_int16.bin'
iq_data = parse.parse_iq_file(filename,np.int16,np.complex64)

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
f_welch,Pxx_welch = plotters.plot_welch_psd(iq_data,nperseg,noverlap,nfft,fs,title_str="Raw IQ Data")

#Plot the Spectrogram
plot_time_downsample_factor=4
f_spec,t_spec,Sxx_spec = plotters.plot_spectrogram(iq_data,nperseg,noverlap,nfft,fs,plot_time_downsample_factor,title_str="Raw IQ Data")

# Plot the magnitude response averaged over blocks
t_block,mag_block = plotters.plot_block_magnitude(iq_data,nperseg,noverlap,fs,title_str="Raw IQ Data")

# Observe signals in the following frequency ranges
#Signal 1: [-0.355,-0.194] * Pi
#Signal 2: [-0.182,-0.10]
#Signal 3: [-0.10,0.065]
#Signal 4: [0.094,0.262]
#Signal 5: [0.296,0.439]
# Can use the Parks Mclellan Algorithm

# Filter Requirements for Signal 4
# 40 dB Stopband attenuation
signal_passband = [0.094,0.262]
signal_bw = signal_passband[1] - signal_passband[0] # BW
up_down = Fraction.from_float(signal_bw).limit_denominator(1000) #No rat function in python :(
up_factor = up_down.numerator
down_factor = up_down.denominator
print(f"BW Estimated {signal_bw}, Up: {up_factor}, Down: {down_factor}")
signal_fc = np.mean(signal_passband)
fp = signal_bw/2 # Half sided bandwidth of FIR lowpass filter
trans_width = 1.2 # 30% transition width
fsb = fp * trans_width
numtaps = 109
bands   = [0.0, fp, fsb, 0.5]
desired = [1, 0]
weight  = [10, 100] 
h_lp = scipy.signal.remez(
    numtaps,
    bands,
    desired,
    weight=weight,
    fs=1.0
)
h_bp = demod.apply_frequency_shift(h_lp,-signal_fc,fs)
w_bp,H_bp = plotters.plot_freqz(h_bp,title_str="Bandpass Filter")
plotters.plot_fft_magnitude_phase(h_bp,nfft=1024,title_str="Bandpass Filter")
# plotters.plot_group_delay(w_bp,H_bp,title_str="Bandpass Filter")
h_group_delay = plotters.plot_group_delay(h_bp,1,title_str="Bandpass Filter")

# Apply Filter
signal_filtered = scipy.signal.lfilter(h_bp,1,iq_data)
f_welch_signal,Pxx_welch_signal = plotters.plot_welch_psd(signal_filtered,nperseg,noverlap,nfft,fs,"Filtered Signal")
t_block_signal,mag_block_signal = plotters.plot_block_magnitude(signal_filtered,nperseg,noverlap,fs,"Filtered Signal")

# Baseband Signal
signal_basebanded = demod.apply_frequency_shift(signal_filtered,signal_fc,fs)
f_welch,Pxx_welch = plotters.plot_welch_psd(signal_basebanded,nperseg,noverlap,nfft,fs,"Signal Basebanded (Post Filter)")
plotters.plot_real_imag(signal_basebanded,fs,title_str="Signal Basebanded")

# Downsample Signal
signal_downsampled = scipy.signal.resample_poly(signal_basebanded, up_factor, down_factor)
plotters.plot_real_imag(signal_downsampled,fs*up_factor/down_factor,title_str="Signal Resampled")
breakpoint()