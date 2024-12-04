# audio_analyzer.py

import librosa
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


class AudioAnalyzer:
    
    def __init__(self, filename, fft_size=256, input_sr=41000, analyze=True):
        """
        Loads in and transform sample data from an audio file into a pandas dataframe for analysis and graphing.
        
        Input: 
            filename: relative path to audio file
            fft_size: The number of frequency bins for the fft analysis. Defaults to 256
            input_sr: the sample rate of the input audio file. Defaults to 41000
        """
        y, sr = librosa.load(filename, sr=input_sr)
        self.y = y
        self.sr = sr
        self.fft_size = fft_size
        if analyze:
            self.spectrum_analysis()
        
    def change_fft_bin_size(self, size):
        """
        Optionally changes the number of fft bins after initialization.
        """
        self.fft_size = size
        self.spectrum_analysis()
        
    def spectrum_analysis(self):
        """
        Runs spectrum analysis on the input audio file. Sets and returns a dataframe with frequency information across all fft bins.
        Amplitudes are averaged at each frequency.
        """
        self.df = pd.DataFrame(np.abs(librosa.stft(self.y, n_fft=self.fft_size)))
        
        bins = librosa.fft_frequencies(sr=self.sr, n_fft=self.fft_size)
        
        self.df['bins'] = bins / 1000. # divide by 1000 lets us display in kHz
        
        self.df['average_amplitude'] = self.df.mean(axis=1)
        self.df = self.df[['bins', 'average_amplitude']]
        return self.df
    
    def plot_spectrum(self, min_freq=0, max_freq=None, fill=False, title="Spectrogram - Average Frequency"):
        """
        Plots a single spectrogram of averaged frequencies across all fft bins. Uses the generated dataframe as the source.
        """
        max_freq = max_freq or 20000.
        window = self.df.loc[(self.df.bins * 1000. >= min_freq) & (self.df.bins * 1000. <= max_freq)].copy()
        window['scaled_amplitude'] = np.interp(window.average_amplitude, (0., max(window.average_amplitude)), (0., 1.))
        window.plot(x='bins', y='scaled_amplitude', figsize=(16,4))
        if fill:            
            plt.fill_between(self.df.bins, self.df.average_amplitude)
        
        legend = plt.legend()
        legend.remove()
        plt.xlabel("Frequency (kHz)", fontsize=20)
        plt.ylabel("Amplitude (scaled)", fontsize=20)
        plt.title(title, fontsize=26)
