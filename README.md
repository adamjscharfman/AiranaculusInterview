1. Load the signal as IQ data, making sure you understand the data format
2. Plot a PSD of the signal
3. Plot a Spectrogram of the signal
4. Plot the magnitude of the signal as a time series (you can average blocks or average or a
moving window to make the plot easier to understand)
5. From the PSD, manually identify a specific band that isolates a signal
1. Filter out all other signals
2. Bring this signal approximately to baseband
3. Decimate the signal by some factor so that the sampling rate is approximately equal
to, or at least close to, the bandwidth of the isolated signal

Please write code to demodulate the simulated Rectangular BPSK signal:
rectangular_bpsk_sps_10_complex_float32.bin
Each symbol is 10 samples long
The first symbol starts at sample 0
Each pulse is rectangular
Extra: Estimate the SNR of the signal