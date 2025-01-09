 Define binary key profiles (assuming profiles are precomputed)
# maj_profile and min_profile should be arrays or lists representing key profiles
pearson_coeff_maj = np.corrcoef(chroma_vals, maj_profile)[0, 1]
pearson_coeff_min = np.corrcoef(chroma_vals, min_profile)[0, 1]

print("Pearson coefficients (major):", pearson_coeff_maj)  # Debug output

# Define a dictionary associating keys to the correlation coefficients
key_dict = {}
keys = np.random.rand(12)  # TODO: Replace with actual key values
print("Keys (for debug):", keys)

for i, key in enumerate(keys):
    key_dict[key] = pearson_coeff_maj  # Note: Replace with the correct correlation values as needed

print("Key dictionary:", key_dict)


import librosa
import librosa.display
import matplotlib.pyplot as plt

def compute_chromagram(x, Fs, chroma_type='stft', print_chromagram=True):
    """
    Compute the chromagram according to the chroma_type

    Args:
        x: input audio file .wav
        Fs: sampling frequency
        chroma_type: type of the chromagram (default='stft')
        print_chromagram: if True, print chromagram (default=True)
        
    Returns:
        chromagram: computed chromagram
    """
    N = 1024   # STFT length

    # Compute the chromagram stft or cqt
    if chroma_type == 'stft': 
        chromagram = librosa.feature.chroma_stft(y=x, sr=Fs, n_chroma=12, n_fft=N)
    elif chroma_type == 'cqt':
        chromagram = librosa.feature.chroma_cqt(y=x, sr=Fs)
    else: 
        print_chromagram = False
    
    # Plot the chromagram
    if print_chromagram: 
        fig, ax = plt.subplots(nrows=1)
        # notes labels
        ax.label_outer()
        librosa.display.specshow(chromagram, y_axis='chroma', x_axis='time', ax=ax)
        if chroma_type == 'stft': 
            ax.set(title='chroma_stft')    
        elif chroma_type == 'cqt':
            ax.set(title='chroma_cqt')
        # color bar
        img = librosa.display.specshow(chromagram, y_axis='chroma', x_axis='time', ax=ax)
        fig.colorbar(img, ax=ax)
        
    return chromagram



def detect_key(C, maj_profile, min_profile, verbose=False):
    """
    Compute an estimate of the musical key given a chromagram.

    Args:
        C: chromagram (2D array or list of chroma values)
        maj_profile: profile for major scales
        min_profile: profile for minor scales
        verbose: if True, print prominence of chromas and correlation coefficients (default=False)

    Returns:
        best_key: best estimate of musical key
        best_corr: correlation coefficient of best_key
        alt_key: alternative estimate of musical key
        alt_best_corr: correlation coefficient of alt_key
    """
    
    # Compute prominence and correlation coefficients
    # TODO: Add actual implementation here

    if verbose:
        print("Prominence of chromas:")
        # TODO: Print computed prominence

        print("Correlation coefficients for major/minor keys:")
        # TODO: Print computed correlation coefficients

    return best_key, best_corr, alt_key, alt_best_corr


# Example usage (assuming appropriate data is provided)
chroma_vals = np.array([0.2, 0.3, 0.5, 0.7, 0.1, 0.9, 0.8, 0.4, 0.6, 0.3, 0.5, 0.2])
maj_profile = np.array([1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0])
min_profile = np.array([0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0])

best_key, best_corr, alt_key, alt_best_corr = detect_key(chroma_vals, maj_profile, min_profile, verbose=True)


