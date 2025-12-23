import numpy as np
import scipy
import matplotlib.pyplot as plt
import frequency_response
import magnitude_response

def plot_real_imag(iq_data:np.array,fs=1.0,title_str:str=None):
    fig_title = "IQ Data"
    if title_str is not None:
        fig_title = f"{title_str} IQ Data"

    t = np.arange(0,len(iq_data),1)/fs
    fig = plt.figure()
    plt.plot(t,iq_data.real,label='real')
    plt.plot(t,iq_data.imag,label='imag')
    plt.xlabel('Time (s)')
    plt.ylabel('Amplitude')
    plt.title(f'{fig_title}')
    plt.legend()
    plt.grid(True)
    plt.show(block=False)
    return

def plot_fft_magnitude(iq_data:np.array,fs:float=1.0,nfft:int=None,title_str:str=None):
    if nfft is None:
        nfft = len(iq_data)
    
    fig_title = "FFT Magnitude"
    if title_str is not None:
        fig_title = f"{title_str} FFT Magnitude"

    f = np.arange(-fs/2,fs/2,fs/nfft)
    iq_fft = np.fft.fftshift(np.fft.fft(iq_data,nfft))
    fig = plt.figure()
    plt.plot(f,np.abs(iq_fft))
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Magnitude')
    plt.title(f"{fig_title}")
    plt.grid(True)
    plt.show(block=False)

    return f,iq_fft

def plot_fft_magnitude_phase(iq_data:np.array,fs:float=1.0,nfft:int=None,title_str:str=None):
    if nfft is None:
        nfft = len(iq_data)
    
    f = np.arange(-fs/2,fs/2,fs/nfft)
    iq_fft = np.fft.fftshift(np.fft.fft(iq_data,nfft))
    fig,ax = plt.subplots(2,1)
    ax[0].plot(f,np.abs(iq_fft))
    ax[0].set_xlabel('Frequency (Hz)')
    ax[0].set_ylabel('Magnitude')
    ax[0].set_title('FFT Magnitude Response')
    ax[0].grid(True)

    ax[1].plot(f,np.angle(iq_fft))
    ax[1].set_xlabel('Frequency (Hz)')
    ax[1].set_ylabel('Phase (rad)')
    ax[1].set_title('FFT Phase Response')
    ax[1].grid(True)
    ax[1].sharex(ax[0])

    if title_str is not None:
        fig.suptitle(f"{title_str}")

    plt.show(block=False)

    return f,iq_fft

def plot_freqz(h_fir:np.array,nfft:int=None,title_str:str=None):
    # Compute frequency response
    if nfft is None:
        nfft=len(h_fir)

    w, H = scipy.signal.freqz(h_fir, worN=nfft)  # w in radians/sample

    # Convert frequency to normalized 0..1 (cycles/sample)
    f = w / (2*np.pi)

    # Magnitude (linear or dB)
    magnitude = np.abs(H)
    magnitude_dB = 20*np.log10(np.abs(H)+1e-12)  # avoid log(0)

    # Phase
    phase = np.angle(H)

    # Plot magnitude
    fig,ax = plt.subplots(2,1)
    # plt.plot(f, magnitude, label='Linear magnitude')
    ax[0].plot(f, magnitude_dB, label='Magnitude (dB)')
    ax[0].set_xlabel('Normalized frequency [cycles/sample]')
    ax[0].set_ylabel('Magnitude')
    ax[0].set_title('FIR Filter Magnitude Response')
    ax[0].grid(True)
    ax[0].legend()

    # Plot phase
    ax[1].plot(f, phase)
    ax[1].set_xlabel('Normalized frequency [cycles/sample]')
    ax[1].set_ylabel('Phase [radians]')
    ax[1].set_title('FIR Filter Phase Response')
    ax[1].sharex(ax[0])
    ax[1].grid(True)

    if title_str is not None:
        fig.suptitle(f"{title_str}")

    plt.show(block=False)

    return w,H

def plot_welch_psd(iq_data:np.array,nperseg:int=1024,noverlap:int=512,nfft:int=1024,fs:float=1.0,title_str:str=None):
    
    fig_title = "Welch PSD"
    if title_str is not None:
        fig_title = f"{title_str} Welch PSD"
    f,Pxx=frequency_response.compute_welch_psd(iq_data,nperseg,noverlap,nfft,fs)
    fig = plt.figure()
    plt.plot(np.fft.fftshift(f), 10*np.log10(np.fft.fftshift(Pxx)))
    plt.xlabel('Frequency [Hz]')
    plt.ylabel('PSD [dBW/Hz]') # Units depend on the scaling parameter
    plt.title(f"{fig_title}")
    plt.grid(True)
    plt.show(block=False)

    return f,Pxx

def plot_spectrogram(iq_data:np.array,nperseg:int=1024,noverlap:int=512,nfft:int=1024,fs:float=1.0,plot_time_downsample:int=1,title_str:str=None):
    
    fig_title = "Spectrogram"
    if title_str is not None:
        fig_title = f"{title_str} Spectrogram"
    f,t,Sxx = frequency_response.compute_spectrogram(iq_data,nperseg,noverlap,nfft,fs)
    fig = plt.figure()
    plt.pcolormesh(t[0::plot_time_downsample], f, Sxx[:,0::plot_time_downsample], shading='nearest')
    plt.title(f"{fig_title}")
    plt.ylabel('Frequency (Hz)')
    plt.xlabel('Time (s)')
    plt.show(block=False)

    return f,t,Sxx

def plot_block_magnitude(iq_data:np.array,blocksize:int=1024,hop:int=512,fs:float=1.0,title_str:str=None):
    t_block,mag_block = magnitude_response.compute_block_magnitude(iq_data,blocksize,hop,fs)
    fig = plt.figure()
    plt.plot(t_block,mag_block)
    plt.xlabel('Time (s)')
    plt.ylabel('Magnitude')
    plt.title(f'Overlapping Block Average: Block Size {blocksize}, Hop Size {hop}')
    plt.grid(True)
    if title_str is not None:
        fig.suptitle(f"{title_str}")
    plt.show(block=False)

    return t_block,mag_block

def plot_constellation(iq_data:np.array,title_str:str=None):

    fig_title = "Constellation Diagram"
    if title_str is not None:
        fig_title = f"{title_str} Constellation Diagram"
    fig = plt.figure()
    plt.scatter(iq_data.real,iq_data.imag)
    plt.title(f"{fig_title}")
    plt.xlabel('Real')
    plt.ylabel('Imag')
    plt.grid(True)
    plt.show(block=False)
    return

def plot_bpsk_pulse_timings(iq_data:np.array,fs:int,samples_per_symbol:int,offset:int=0,title_str:str=None):
    
    fig_title = "Optimal Sample Times"
    if title_str is not None:
        fig_title = f"{title_str} Optimal Sample Times"
    t = np.arange(0,len(iq_data),1)/fs
    t_timing = t[samples_per_symbol//2+offset::samples_per_symbol]
    iq_data_timing = iq_data[samples_per_symbol//2+offset::samples_per_symbol]
    fig = plt.figure()
    plt.plot(t,iq_data.real,label='BPSK Real')
    plt.scatter(t_timing,iq_data_timing.real,marker='o',label='Sample Timings')
    plt.xlabel('Time (s)')
    plt.ylabel('Amplitude')
    plt.title(f"{fig_title}")
    plt.legend()
    plt.grid(True)
    plt.show(block=False)
    return

def plot_group_delay(w:np.array,H:np.array,title_str:str=None):

    fig_title = "Group Delay Response"
    if title_str is not None:
        fig_title = f"{title_str} Group Delay Response"
    gd = frequency_response.compute_group_delay(w,H)
    plt.figure()
    plt.plot(w[:-1], gd)
    plt.xlabel("Frequency (rad/sample)")
    plt.ylabel("Group Delay (samples)")
    plt.title(f"{fig_title}")
    plt.grid(True)
    plt.show(block=False)
    return gd
    
