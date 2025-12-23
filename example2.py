import sys,os
sys.path.append('lib')
import parse
import plotters
import numpy as np
import matplotlib.pyplot as plt
import scipy
import demod

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
# Signal 1: [0.14,0.273]
# Signal 2: [-0.05,-0.36]
# Can use the Parks Mclellan Algorithm

# Approach 1: Design a Lowpass FIR filter and shift it to the passband of the signal
# numtaps = 100
numtaps = 201      # Size of the FIR filter.
signal_1_passband = [0.136,0.241]
# signal_1_passband = [0.131,0.245]
signal_1_bw = signal_1_passband[1] - signal_1_passband[0]
signal_1_fc = np.sum(signal_1_passband)/2
# Prototype cutoff = half bandwidth
fp = signal_1_bw / 2
fsb = fp * 1.3   # 10% transition band (reasonable)
bands = [0.0, fp, fsb, 0.5]
desired = [1, 0]
# weight = [10,1]
h_lp = scipy.signal.remez(
    numtaps,
    bands,
    desired,
    # weight=weight,
    fs=1.0
)
h_bp = demod.apply_frequency_shift(h_lp,-signal_1_fc,fs) #Shift Lowpass filter to passband of signal
w_bp,H_bp = plotters.plot_freqz(h_bp)
plotters.plot_fft_magnitude_phase(h_bp,1024)
plotters.plot_group_delay(w_bp,H_bp)
# Apply Filter
signal_1_filtered = scipy.signal.lfilter(h_bp,1,iq_data)
f_welch_signal1,Pxx_welch_signal1 = plotters.plot_welch_psd(signal_1_filtered,nperseg,noverlap,nfft,fs)
t_block_signal1,mag_block_signal1 = plotters.plot_block_magnitude(signal_1_filtered,nperseg,noverlap,fs)

# Approach 2: Design a bandtop filter at the interference to isolate the signal
# Signal 2 at [-0.36, -0.06]
numtaps = 201
signal_2_passband = [-0.36, -0.06]
signal_2_bw = signal_2_passband[1] - signal_2_passband[0]
signal_2_fc = np.sum(signal_2_passband)/2
signal_2_fp = signal_2_bw / 2 #Passband (single sided)
signal_2_fsb = signal_2_fp * 1.1   # 10% transition band (reasonable)
signal_2_bands = [0.0, signal_2_fp, signal_2_fsb, 0.5]
signal_2_desired = [1, 0]
# Generate Lowpass Filter for Bandwidth of signal
h_lp = scipy.signal.remez(
    numtaps,
    signal_2_bands,
    signal_2_desired,
    # weight=weight,
    fs=1.0
)
# Shift lowpass filter to desired passband (stopband)
h_bp = demod.apply_frequency_shift(h_lp,-signal_2_fc,fs)
# Compute bandstop filter by subtracting bandpass filter
h_bs = np.zeros(numtaps, dtype=complex)
h_bs[(numtaps - 1)//2] = 1.0    # delayed delta = all-pass
h_bs -= h_bp
w_bs,H_bs = plotters.plot_freqz(h_bs)
plotters.plot_fft_magnitude_phase(h_bs,1024)

# Apply Filter
signal_2_filtered = scipy.signal.lfilter(h_bs,1,iq_data)
f_welch_signal2,Pxx_welch_signal2 = plotters.plot_welch_psd(signal_2_filtered,nperseg,noverlap,nfft,fs)
t_block_signal2,mag_block_signal2 = plotters.plot_block_magnitude(signal_2_filtered,nperseg,noverlap,fs)


breakpoint()