from scipy import signal
import numpy as np

def compute_welch_psd(iq_data:np.array,nperseg:int=1024,noverlap:int=512,nfft:int=1024,fs:float=1.0)->tuple[np.array,np.array]:
    f,Pxx = signal.welch(iq_data,fs=fs,nperseg=nperseg,noverlap=noverlap,nfft=nfft)
    return f,Pxx

def compute_spectrogram(iq_data:np.array,nperseg:int=1024,noverlap:int=512,nfft:int=1024,fs:float=1.0)->tuple[np.array,np.array,np.array]:
    f, t, Sxx = signal.spectrogram(iq_data,nperseg=nperseg,noverlap=noverlap,nfft=nfft,fs=fs,return_onesided=False,mode='magnitude')
    return np.fft.fftshift(f),t,np.fft.fftshift(Sxx,axes=0)

def compute_group_delay(w:np.array,H:np.array)->np.array:
    phase = np.unwrap(np.angle(H))
    gd = -np.diff(phase) / np.diff(w)    
    return gd

