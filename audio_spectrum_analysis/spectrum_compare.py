# spectrum_compare.py

from audio_analyzer import AudioAnalyzer
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import ttest_ind
import matplotlib.patheffects as path_effects


class SpectrumCompareFour:
    
    def __init__(self, demucs, spleeter, wun, oum):
        """
        Compares the frequency and amplitude information of four AudioAnalyzer class instances.
        
        Input:
            demucs, spleeter, wun, oum: Four AudioAnalyzer class instances.
        """
        # Store all DataFrames
        self.demucs_df = demucs.df
        self.spleeter_df = spleeter.df
        self.wun_df = wun.df
        self.oum_df = oum.df

        # Store as a list for easier iteration
        self.dfs = [self.demucs_df, self.spleeter_df, self.wun_df, self.oum_df]
        
        # Calculate max average amplitude and scale amplitudes
        self.get_max_average()
        self.scale_amplitudes()
        self.get_ratio_dfs()
    
    def get_max_average(self):
        max_average = 0
        for df in self.dfs:
            cur_max = df.average_amplitude.max()
            max_average = max(cur_max, max_average)
        
        self.max_average = max_average

    def scale_amplitudes(self):
        """
        Scales the amplitude for each frequency bin to make comparisons consistent across all four instances.
        """
        scaled_dfs = []
        for df in self.dfs:
            averaged = df['average_amplitude']
            df['scaled_amplitude'] = np.interp(averaged, (0., self.max_average), (0., 1.))
            scaled_dfs.append(df)
        self.demucs_df, self.spleeter_df, self.wun_df, self.oum_df = scaled_dfs

    def get_ratio_dfs(self):
        """
        Computes the difference in scaled amplitudes for each pair of audio files.
        """
        self.ratio_dfs = []
        for i in range(1, len(self.dfs)):
            ratio_df = pd.DataFrame((self.dfs[i].scaled_amplitude - self.demucs_df.scaled_amplitude) + 0.5)
            ratio_df['bins'] = self.demucs_df.bins
            ratio_df.loc[ratio_df.scaled_amplitude > 1.0, ['scaled_amplitude']] = 1.0
            ratio_df.loc[ratio_df.scaled_amplitude < 0.0, ['scaled_amplitude']] = 0.0
            self.ratio_dfs.append(ratio_df)

    def plot_spectrum_group(self,
                            title="Comparison of Frequency Amplitudes",
                            xlabel="Frequency (kHz)",
                            ylabel="Scaled Amplitude",                             
                            frange=None,
                            legend=["demucs", "Modified 1", "Modified 2", "Modified 3"],
                           ):  
        """
        Plots a spectrogram comparing the frequencies and relative amplitudes at each fft bin of the four AudioAnalyzer class instances.
        """
        fig = plt.figure(figsize=(16, 8))
        
        # Iterate through all audio DataFrames to plot
        for i, df in enumerate(self.dfs):
            if frange:
                df = df.loc[(df.bins * 1000. >= frange[0]) & (df.bins * 1000. <= frange[1])]
            plt.plot(df.bins, df.scaled_amplitude, label=legend[i])
        
        plt.title(title, fontsize=24)
        plt.xlabel(xlabel, fontsize=20)
        plt.ylabel(ylabel, fontsize=20)
        plt.legend(fontsize=14)

    def plot_amplitude_distributions(self, 
                                     size=1000,
                                     title="Amplitude Distributions",
                                     xlabel="Average Amplitude (unscaled)",
                                     ylabel="Density"):
        """
        Plots the distributions of amplitudes for all four instances to help identify the cleanest audio.
        """
        plt.figure(figsize=(10, 8))
        
        for i, df in enumerate(self.dfs):
            amp = df.scaled_amplitude
            samples = [np.random.choice(amp, size=size).mean() for _ in range(size)]
            sns.distplot(samples, label=f"Audio {i+1}")

        plt.title(title, fontsize=18)
        plt.xlabel(xlabel, fontsize=14)
        plt.ylabel(ylabel, fontsize=14)
        plt.legend(fontsize=12)
        
        # Statistical comparison between all audios
        stats_results = []
        for i in range(1, len(self.dfs)):
            t_stat, p_val = ttest_ind(self.dfs[0].scaled_amplitude, self.dfs[i].scaled_amplitude, equal_var=False)
            stats_results.append((f"Audio 1 vs Audio {i+1}", t_stat, p_val))

        return pd.DataFrame(stats_results, columns=["Comparison", "T-Statistic", "P-Value"])
