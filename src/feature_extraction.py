
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA


def extract_fft_features(signal):

    fft_result = np.fft.fft(signal)
    magnitude = np.abs(fft_result)
    phase = np.angle(fft_result)


    features = []
    features.append(np.max(magnitude))
    features.append(np.mean(magnitude))
    features.append(np.std(magnitude))
    features.append(np.sum(magnitude[:len(magnitude) // 2]))
    features.append(np.sum(magnitude[len(magnitude) // 2:]))
    features.append(np.median(phase))
    return np.array(features)


def apply_fft_to_samples(features):

    fft_features = []
    for sample in features:
        sample_fft_features = []

        for i in range(sample.shape[0]):
            signal = np.array([sample[i]])
            fft_feats = extract_fft_features(signal)
            sample_fft_features.extend(fft_feats)
        fft_features.append(np.array(sample_fft_features))
    
    fft_features = np.array(fft_features)
    return fft_features


def normalize_fft_features(fft_features):

    fft_scaler = MinMaxScaler()
    normalized_fft = fft_scaler.fit_transform(fft_features)
    return normalized_fft, fft_scaler


def concatenate_features(original_features, fft_features):

    return np.concatenate([original_features, fft_features], axis=1)


def apply_pca(features, variance_ratio=0.95):

    original_dim = features.shape[1]
    pca = PCA(n_components=variance_ratio)
    reduced_features = pca.fit_transform(features)
    print(f"特征维度从 {original_dim} 降维到 {pca.n_components_}")
    return reduced_features, pca


def frequency_domain_augmentation(features, alpha=0.1):

    fft_features = np.fft.fft(features, axis=1)
    magnitude = np.abs(fft_features)
    phase = np.angle(fft_features)


    magnitude *= (1 + alpha * np.random.randn(*magnitude.shape))
    phase += alpha * np.random.randn(*phase.shape)


    augmented = magnitude * np.exp(1j * phase)
    return np.real(np.fft.ifft(augmented, axis=1))
