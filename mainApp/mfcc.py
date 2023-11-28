import numpy as np
import librosa
import pywt

def prepare_data(fnames):
    audio_length=8000 * 1
    dim = (40, 1 + int(np.floor(audio_length/512)), 1)
    X = np.empty(shape=(len(fnames), dim[0], dim[1], 1))
    input_length = audio_length
    for i, fname in enumerate(fnames):
        data, _ = librosa.core.load(fname, sr=8000, res_type="kaiser_fast")
        # Random offset / Padding
        if len(data) > input_length:
            max_offset = len(data) - input_length
            offset = np.random.randint(max_offset)
            data = data[offset:(input_length+offset)]
        else:
            if input_length > len(data):
                max_offset = input_length - len(data)
                offset = np.random.randint(max_offset)
            else:
                offset = 0
            data = np.pad(data, (offset, input_length - len(data) - offset), "constant")

        # data = np.pad(data, (1, 1), "constant")
        data = librosa.feature.mfcc(y=data, sr=8000, n_mfcc=40)
        data = np.expand_dims(data, axis=-1)
        X[i,] = data
        # X=data
    return X


def prepare_data2(fnames):
    audio_length=8000 * 1
    dim = (60, 1 + int(np.floor(audio_length/512)), 1)
    dim_spectral_centroid = (1, 1 + int(np.floor(audio_length / 512)), 1)
    
    X_mfcc = []
    X_spectral_centroid = []

    input_length = audio_length

    for i, fname in enumerate(fnames):
        data, _ = librosa.core.load(fname, sr=8000, res_type="kaiser_fast")

        # Random offset / Padding
        if len(data) > input_length:
            max_offset = len(data) - input_length
            offset = np.random.randint(max_offset)
            data = data[offset:(input_length + offset)]
        else:
            if input_length > len(data):
                max_offset = input_length - len(data)
                offset = np.random.randint(max_offset)
            else:
                offset = 0
            data = np.pad(data, (offset, input_length - len(data) - offset), "constant")

        # Extract MFCCs
        mfccs = librosa.feature.mfcc(y=data, sr=8000, n_mfcc=60)
        mfccs = np.expand_dims(mfccs, axis=-1)
        X_mfcc.append(mfccs)

        # Extract Spectral Centroid
        spectral_centroid = librosa.feature.spectral_centroid(y=data, sr=8000)
        spectral_centroid = np.expand_dims(spectral_centroid, axis=-1)
        X_spectral_centroid.append(spectral_centroid)

    # Ensure the same time and frequency dimensions for both features
    max_time_dim = max(max(x.shape[1] for x in X_mfcc), max(x.shape[1] for x in X_spectral_centroid))
    max_freq_dim = max(X_mfcc[0].shape[0], X_spectral_centroid[0].shape[0])

    # Pad and reshape the features individually
    X_mfcc_padded = np.array([np.pad(x, ((0, max_freq_dim - x.shape[0]), (0, 0), (0, max_time_dim - x.shape[1])), mode='constant') for x in X_mfcc])
    X_spectral_centroid_padded = np.array([np.pad(x, ((0, max_freq_dim - x.shape[0]), (0, 0), (0, max_time_dim - x.shape[1])), mode='constant') for x in X_spectral_centroid])

    # Combine MFCCs and Spectral Centroid
    X_combined = np.concatenate([X_mfcc_padded, X_spectral_centroid_padded], axis=-1)

    return X_combined

# Mean Absolute Deviation
def maddest(d, axis=None):
    np.random.seed(1337)
    noise = np.random.normal(0, 0.5, 150_000)
    return np.mean(np.absolute(d - np.mean(d, axis)), axis)

# Denoise the raw signal given a segment x
def denoise_signal(x, wavelet='db4', level=1):
    coeff = pywt.wavedec(x, wavelet, mode="per")
    sigma = (1/0.6745) * maddest(coeff[-level])
    uthresh = sigma * np.sqrt(2*np.log(len(x)))
    coeff[1:] = (pywt.threshold(i, value=uthresh, mode='hard') for i in coeff[1:])

    return pywt.waverec(coeff, wavelet, mode='per')

# Denoise the raw signal (simplified) given a segment x
def auto_denoise_signal(x, wavelet='db4', level=1):
    coeff = pywt.wavedec(x, wavelet, mode="per")
    #univeral threshold
    uthresh = 10
    coeff[1:] = (pywt.threshold(i, value=uthresh, mode='hard') for i in coeff[1:])
    # Reconstruct the signal using the thresholded coefficients
    return pywt.waverec(coeff, wavelet, mode='per')

def prepare_data3(fnames):
    audio_length=8000 * 1
    dim_wavelet = (50, 1 + int(np.floor(audio_length/512)), 1)
    
    X = np.empty(shape=(len(fnames), dim_wavelet[0], dim_wavelet[1], 1))
    input_length = audio_length
    for i, fname in enumerate(fnames):
        data, _ = librosa.core.load(fname, sr=8000, res_type="kaiser_fast")
        # Random offset / Padding
        if len(data) > input_length:
            max_offset = len(data) - input_length
            offset = np.random.randint(max_offset)
            data = data[offset:(input_length+offset)]
        else:
            if input_length > len(data):
                max_offset = input_length - len(data)
                offset = np.random.randint(max_offset)
            else:
                offset = 0
            data = np.pad(data, (offset, input_length - len(data) - offset), "constant")

        data = denoise_signal(x=data)
        data = librosa.feature.mfcc(y=data, sr=8000, n_mfcc=50)
        data = np.expand_dims(data, axis=-1)
        X[i,] = data
        # X=data
    return X