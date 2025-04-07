import numpy as np

def add_noise(X, snr_db):
    signal_power = np.mean(X ** 2)
    snr_linear = 10 ** (snr_db / 10)
    noise_power = signal_power / snr_linear
    noise = np.random.normal(0, np.sqrt(noise_power), X.shape)
    return X + noise
