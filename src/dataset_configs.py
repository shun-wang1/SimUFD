from typing import Dict, Any
import os
import numpy as np
import h5py
from scipy.signal import hilbert
import time

DATA_CONFIGS: Dict[str, Dict[str, Any]] = {
    'IMS_2_1': {
        'path': 'IMS',
        'file_name': 'IMS_2_1.mat',
        'sample_length': 20480,
        'window_size': 2048,
        'fs': 20480,
        'channels': {
            'CH1': 0,  # 1
            'CH2': 1,  # 2
            'CH3': 2,  # 3
            'CH4': 3   # 4
        },
        'default_channels': 'CH1',
        'split': {
            'train_timesteps': 80,
            'val_timesteps': 100,
        }
    },
    'XJ_2_2': {
        'path': 'XJ',
        'file_name': 'Bearing2_2.mat',
        'sample_length': 32768,
        'window_size': 2048,
        'fs': 20480,
        'channels': {
            'CH1': 0,
            'CH2': 1
        },
        'default_channels': 'CH1',
        'split': {
            'train_timesteps': 24,
            'val_timesteps': 30,
        }
    },
    'XJ_2_3': {
        'path': 'XJ',
        'file_name': 'Bearing2_3.mat',
        'sample_length': 32768,
        'window_size': 2048,
        'fs': 20480,
        'channels': {
            'CH1': 0,
            'CH2': 1
        },
        'default_channels': 'CH1',
        'split': {
            'train_timesteps': 80,
            'val_timesteps': 100,
        }
    },
    'XJ_2_5': {
        'path': 'XJ',
        'file_name': 'Bearing2_5.mat',
        'sample_length': 32768,
        'window_size': 2048,
        'fs': 20480,
        'channels': {
            'CH1': 0,
            'CH2': 1
        },
        'default_channels': 'CH1',
        'split': {
            'train_timesteps': 80,
            'val_timesteps': 100,
        }
    },
}

class PHMDataProcessor:
    def __init__(self, data_path, sample_length, window_size, fs):
        self.data_path = data_path
        self.sample_length = sample_length
        self.window_size = window_size
        self.step = window_size
        self.fs = fs
        self.fft_mode = 'raw' # 'raw'   'normalized'

    def _calculate_fft(self, signal: np.ndarray):
        fftsize = len(signal)
        fft_values = np.fft.fft(signal)
        freqs = np.fft.fftfreq(fftsize, 1.0 / self.fs)

        # Calculate amplitudes
        if self.fft_mode == 'raw':
            amplitudes = np.abs(fft_values)
        elif self.fft_mode == 'normalized':
            amplitudes = np.abs(fft_values) / fftsize
            if fftsize % 2 == 0:
                amplitudes[1:-1] *= 2
            else:
                amplitudes[1:] *= 2
        else:
            raise ValueError(f"Unsupported FFT mode: {self.fft_mode}")

        # Get positive frequencies only
        positive_indices = freqs >= 0
        return freqs[positive_indices].flatten(), amplitudes[positive_indices].flatten()

    def _process_signal_window(self, signal, method):
        """Process a single signal window using specified method"""
        if method == 'FFT':
            return self._calculate_fft(signal)
        elif method == 'ES':
            envelope = np.abs(hilbert(signal))
            return self._calculate_fft(envelope)
        elif method == 'LogES':
            analytic_signal = hilbert(signal)
            envelope = np.abs(analytic_signal)
            freqs, spectrum = self._calculate_fft(envelope)
            return freqs, np.log1p(spectrum)
        else:
            raise ValueError(f"Unsupported processing method: {method}")

    def load_mat_file(self, filename):
        """Load MAT file data"""
        file_path = self.data_path
        variable_name = os.path.splitext(filename)[0]
        try:
            with h5py.File(file_path, 'r') as mat:
                try:
                    data = np.transpose(mat[variable_name])
                except KeyError:
                    data = np.transpose(mat['text_data'])
        except Exception as e:
            print(f"Error loading MAT file: {e}")
            raise
        return data

    def process_spectrum(self, all_data, channel_index, method):

        channel_data = all_data[:, channel_index]
        num_timesteps = len(channel_data) // self.sample_length

        timestep_list = []
        freqs_list = []
        fft_spectrum_list = []

        spectrum_times = []
        total_windows = 0

        for timestep in range(num_timesteps):
            timestep_start = timestep * self.sample_length

            for i in range(0, self.sample_length - self.window_size + 1, self.step):
                global_start  = timestep_start + i

                signal = channel_data[global_start:global_start + self.window_size]
                start_time = time.time()
                freqs, spectrum = self._process_signal_window(signal, method)
                end_time = time.time()

                spectrum_time = (end_time - start_time) * 1000  # ms
                spectrum_times.append(spectrum_time)
                total_windows += 1

                timestep_list.append(timestep + 1)
                freqs_list.append(freqs)
                fft_spectrum_list.append(spectrum)

        return timestep_list, fft_spectrum_list, freqs_list


def get_data(args):
    try:
        processor = PHMDataProcessor(
            data_path=args.data_path,
            sample_length=args.sample_length,
            window_size=args.window_size,
            fs=args.fs,
        )
        all_data = processor.load_mat_file(args.data_file_name)

        # Process based on model type
        if args.model in ['SimUFD']:
            return processor.process_spectrum(all_data, args.channel_indices,'LogES')
        else:
            raise ValueError(f"Unknown model type: {args.model}")

    except Exception as e:
        raise RuntimeError(f"Error processing data for {args.data_file_name}: {str(e)}")

def set_data_params(args):
    data_config = DATA_CONFIGS[args.data_type]
    args.data_path = os.path.normpath(os.path.join(args.data_dir,
                                                   data_config['path'],
                                                   data_config['file_name']))
    args.data_file_name = data_config['file_name']
    args.sample_length = data_config['sample_length']
    args.window_size = data_config['window_size']
    args.fs = data_config['fs']

    selected_channels = data_config['default_channels']

    args.channel_indices = data_config['channels'][selected_channels]

    args.train_timesteps = data_config['split']['train_timesteps']
    args.val_timesteps = data_config['split']['val_timesteps']

