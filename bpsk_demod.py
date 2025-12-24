import sys,os
sys.path.append('lib')
import parse
import plotters
import demod
import numpy as np
import matplotlib.pyplot as plt


filename = 'data/rectangular_bpsk_sps_10_complex_float32.bin'
iq_data = parse.parse_iq_file(filename,np.float32,np.complex64)

# Plot the Signal from the file
plotters.plot_real_imag(iq_data,title_str = "Raw IQ Data")

# Plot the Constellation of the BPSK
plotters.plot_constellation(iq_data,title_str='Raw Data')

# Plot the FFT from the file
f_fft,mag_fft = plotters.plot_fft_magnitude_phase(iq_data, title_str = 'Raw IQ Data')

# Plot the Welch PSD
fs = 1.0 #Not specified, just using a nominal value of 1
nperseg = 512 # Number of samples per frequency response
# nperseg = 1024
# nperseg = 4094
# nperseg = 2048
# noverlap = nperseg//2 #50% Overlap
noverlap = nperseg//4 #25% Overlap
nfft = nperseg # No zero padding on the fft
f_welch,Pxx_welch = plotters.plot_welch_psd(iq_data,nperseg,noverlap,nfft,fs, title_str = "Raw IQ Data")
# Note data appears to be centered at baseband, no need to complex baseband

# Estimate Frequency Offset
samples_per_symbol=10
f_offset_est = demod.estimate_bpsk_frequency_offset_one_lag_estimator(iq_data,samples_per_symbol)
print(f"Estimated Normalized Frequency Offset: {f_offset_est:0.3e}")

# Correct Frequency Offset
t_elapsed = np.arange(len(iq_data))
iq_data_baseband = iq_data * np.exp(-1j*2*np.pi*f_offset_est*t_elapsed/fs)
f_welch_baseband,Pxx_welch_baseband = plotters.plot_welch_psd(iq_data_baseband,nperseg,noverlap,nfft,fs, title_str = "Basebanded IQ Data")

# Filter signal with a matched filter - Optimal detector is a matched filter
h = np.ones(samples_per_symbol)/samples_per_symbol
iq_matched_filt = np.convolve(iq_data,h,mode='same')
nfft_match = 2048
f_match,fft_match = plotters.plot_fft_magnitude_phase(h,fs,nfft_match,title_str = "Matched Filter")
w_matched,H_matched = plotters.plot_freqz(h,nfft_match,"Matched Filter")
# plotters.plot_real_imag(iq_matched_filt)
plotters.plot_group_delay(w_matched,H_matched,"Matched Filter")

# Most SNR at Center bin (Triangle peak)
plotters.plot_real_imag(iq_matched_filt)
iq_demod_timings = iq_matched_filt[samples_per_symbol//2::samples_per_symbol]
plotters.plot_bpsk_pulse_timings(iq_matched_filt,fs,samples_per_symbol,0)
# plotters.plot_real_imag(iq_demod_timings)
plotters.plot_constellation(iq_demod_timings,title_str='Filtered Data')

# Compute average rotation
num_bits_per_symbol = 1 
phi_hat = demod.estimate_m_psk_rotation(iq_demod_timings,num_bits_per_symbol)
iq_demod_timings_rot = iq_demod_timings * np.exp(-1j*phi_hat)
plotters.plot_constellation(iq_demod_timings_rot,title_str="Rotated")
print(f"Estimated Rotation: {np.degrees(phi_hat)} deg")

# Demodulate Signal - Only need the real part if constellation is not rotated
demod_bits = np.real(iq_demod_timings) >= 0
demod_amplitudes = 2*demod_bits.astype(np.float32) - 1

#Estimate SNR
P_sig = 1 # Abs(V**2)
P_noise = np.mean(np.abs(iq_demod_timings - demod_amplitudes)**2) #Variance sigma^2 of the noise is the noise power
BPSK_SNR_db = 10*np.log10(P_sig/P_noise)
print(f"BPSK Signal Estimated SNR Post Matched Filter: {BPSK_SNR_db} dB")
MF_Gain_db = 10*np.log10(samples_per_symbol)
print(f"BPSK Signal Estimated SNR: {BPSK_SNR_db-MF_Gain_db} dB")
breakpoint()