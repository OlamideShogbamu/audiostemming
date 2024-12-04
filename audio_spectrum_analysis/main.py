import os
from google.colab import drive
from audio_analyzer import AudioAnalyzer
from spectrum_compare import SpectrumCompareFour
import matplotlib.pyplot as plt

# Set the path to your Google Drive root directory
drive.mount('/content/drive')
google_drive_path = "/path/to/your/google/drive"

# Define the paths to each model's output folder
models = ['Demucs_Output', 'OpenUnmix_Output', 'Spleeter_Output', 'WaveUNet_Output']
model_paths = {model: os.path.join(google_drive_path, model) for model in models}

# Number of files to process (from _1 to _36)
num_files = 36

# Iterate over each numbered audio file (_1, _2, ..., _36)
for i in range(1, num_files + 1):
    file_suffix = f"_{i}.wav"  # Assuming the files are WAV files

    # Collect paths for each model output for the specific file number
    try:
        audio_files = [
            AudioAnalyzer(os.path.join(model_paths['Demucs_Output'], f"demucs_output{file_suffix}")),
            AudioAnalyzer(os.path.join(model_paths['OpenUnmix_Output'], f"openunmix_output{file_suffix}")),
            AudioAnalyzer(os.path.join(model_paths['Spleeter_Output'], f"spleeter_output{file_suffix}")),
            AudioAnalyzer(os.path.join(model_paths['WaveUNet_Output'], f"waveunet_output{file_suffix}"))
        ]
        
        # Create an instance of SpectrumCompareFour with the loaded audio files
        comparison = SpectrumCompareFour(*audio_files)
        
        # Plot and save the spectrum comparison
        plt.figure()
        comparison.plot_spectrum_group(title=f"Frequency Comparison for File {file_suffix}")
        plt.savefig(os.path.join(google_drive_path, f"frequency_comparison{file_suffix}.png"))

        # Plot and save the amplitude distributions
        plt.figure()
        comparison.plot_amplitude_distributions(title=f"Amplitude Distribution for File {file_suffix}")
        plt.savefig(os.path.join(google_drive_path, f"amplitude_distribution{file_suffix}.png"))

        # Plot and save the heatmap comparison
        plt.figure()
        comparison.plot_spectrum_heatmap(title=f"Heatmap Comparison for File {file_suffix}", cmap="plasma")
        plt.savefig(os.path.join(google_drive_path, f"heatmap_comparison{file_suffix}.png"))
        
        print(f"Plots for file {file_suffix} saved successfully.")
        
    except Exception as e:
        print(f"Error processing file {file_suffix}: {e}")
