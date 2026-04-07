"""
EEG Preprocessing functions for microstate analysis.

This module provides preprocessing functions commonly used in EEG microstate analysis,
including filtering, rereferencing, artifact rejection, and epoching.

These functions mirror functionality from EEGLAB's preprocessing pipeline.
"""

import numpy as np
from scipy import signal
from scipy.linalg import solve
from typing import Tuple, Optional, Union
from dataclasses import dataclass

from eeg_loader import EEGData


def bandpass_filter(
    data: np.ndarray,
    srate: float,
    lowcut: float = 2.0,
    highcut: float = 30.0,
    order: int = 4,
    filttype: str = 'bandpass'
) -> np.ndarray:
    """
    Apply a Butterworth bandpass filter to EEG data.
    
    This implements EEGLAB's pop_eegfiltnew function.
    
    Parameters
    ----------
    data : np.ndarray
        EEG data with shape (n_channels, n_timepoints)
    srate : float
        Sampling rate in Hz
    lowcut : float
        Lower cutoff frequency in Hz (default: 2.0)
    highcut : float
        Upper cutoff frequency in Hz (default: 30.0)
    order : int
        Filter order (default: 4)
    filttype : str
        Filter type: 'bandpass', 'lowpass', or 'highpass'
    
    Returns
    -------
    np.ndarray
        Filtered EEG data with same shape as input
    
    Examples
    --------
    >>> data = np.random.randn(32, 5000)  # 32 channels, 5000 timepoints
    >>> srate = 250.0
    >>> filtered = bandpass_filter(data, srate, lowcut=2.0, highcut=30.0)
    >>> print(filtered.shape)  # (32, 5000)
    """
    nyquist = srate / 2.0
    
    # Normalize frequencies
    if filttype == 'bandpass':
        low = lowcut / nyquist
        high = highcut / nyquist
        b, a = signal.butter(order, [low, high], btype='bandpass')
    elif filttype == 'lowpass':
        high = highcut / nyquist
        b, a = signal.butter(order, high, btype='lowpass')
    elif filttype == 'highpass':
        low = lowcut / nyquist
        b, a = signal.butter(order, low, btype='highpass')
    else:
        raise ValueError(f"Unknown filttype: {filttype}")
    
    # Apply filter (using filtfilt for zero-phase distortion)
    # Process each channel separately
    if data.ndim == 1:
        filtered = signal.filtfilt(b, a, data)
    else:
        n_channels = data.shape[0]
        filtered = np.zeros_like(data)
        for ch in range(n_channels):
            filtered[ch] = signal.filtfilt(b, a, data[ch])
    
    return filtered


def average_reference(data: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Apply average referencing to EEG data.
    
    This computes the average of all channels and subtracts it from
    each channel, which is the standard average reference.
    
    Parameters
    ----------
    data : np.ndarray
        EEG data with shape (n_channels, n_timepoints)
    
    Returns
    -------
    Tuple[np.ndarray, np.ndarray]
        - Referenced EEG data
        - Reference signal (the average across channels)
    
    Examples
    --------
    >>> data = np.random.randn(32, 5000)
    >>> ref_data, ref = average_reference(data)
    >>> print(ref_data.shape)  # (32, 5000)
    """
    # Compute average reference across channels
    reference = np.mean(data, axis=0, keepdims=True)
    
    # Subtract reference from each channel
    referenced = data - reference
    
    return referenced, reference


def reference_to_carbon(
    data: np.ndarray,
    carbon_channels: list
) -> np.ndarray:
    """
    Re-reference EEG data to a specific carbon (channel) or channel group.
    
    Parameters
    ----------
    data : np.ndarray
        EEG data with shape (n_channels, n_timepoints)
    carbon_channels : list
        List of channel indices to use as reference
    
    Returns
    -------
    np.ndarray
        Re-referenced EEG data
    """
    if not carbon_channels:
        # If no carbon specified, use average reference
        ref_data, _ = average_reference(data)
        return ref_data
    
    # Get carbon channel(s)
    carbon_data = data[carbon_channels, :]
    
    # Average of carbon channels
    reference = np.mean(carbon_data, axis=0, keepdims=True)
    
    # Subtract reference
    referenced = data - reference
    
    return referenced


def joint_probability_rejection(
    data: np.ndarray,
    max_channels: int = None,
    threshold: float = 3.0,
    replace_with_nan: bool = False
) -> np.ndarray:
    """
    Reject artifacts using joint probability.
    
    This implements EEGLAB's pop_jointprob function. It identifies
    time points where channels deviate significantly from their expected
    distribution (based on kurtosis).
    
    Parameters
    ----------
    data : np.ndarray
        EEG data with shape (n_channels, n_timepoints)
    max_channels : int, optional
        Maximum number of channels that can be rejected at any time point.
        If None, use all channels.
    threshold : float
        Threshold in standard deviations (default: 3.0)
    replace_with_nan : bool
        If True, replace rejected samples with NaN. If False, interpolate.
    
    Returns
    -------
    np.ndarray
        Data with artifacts marked/rejected
    
    Notes
    -----
    This function calculates the kurtosis of each channel's distribution
    and identifies time points where channels have unusually high or low values
    (beyond threshold standard deviations from the mean).
    """
    n_channels, n_timepoints = data.shape
    
    if max_channels is None:
        max_channels = n_channels
    
    # Calculate z-scores for each channel
    mean = np.mean(data, axis=1, keepdims=True)
    std = np.std(data, axis=1, keepdims=True)
    
    # Avoid division by zero
    std = np.where(std == 0, 1, std)
    
    z_scores = np.abs((data - mean) / std)
    
    # Identify "bad" time points (where too many channels exceed threshold)
    bad_timepoints = np.sum(z_scores > threshold, axis=0) > max_channels
    
    # Mark bad time points
    if replace_with_nan:
        result = data.copy()
        result[:, bad_timepoints] = np.nan
    else:
        # Interpolate bad time points
        result = data.copy()
        for t in np.where(bad_timepoints)[0]:
            # Simple interpolation from neighboring time points
            if t > 0 and t < n_timepoints - 1:
                result[:, t] = (result[:, t-1] + result[:, t+1]) / 2
            elif t > 0:
                result[:, t] = result[:, t-1]
            elif t < n_timepoints - 1:
                result[:, t] = result[:, t+1]
    
    return result


def create_epochs(
    data: np.ndarray,
    srate: float,
    epoch_duration: float = 1.0,
    overlap: float = 0.0
) -> np.ndarray:
    """
    Create epochs from continuous EEG data.
    
    This implements EEGLAB's eeg_regepochs function.
    
    Parameters
    ----------
    data : np.ndarray
        Continuous EEG data with shape (n_channels, n_timepoints)
    srate : float
        Sampling rate in Hz
    epoch_duration : float
        Duration of each epoch in seconds (default: 1.0)
    overlap : float
        Overlap between epochs in seconds (default: 0.0)
    
    Returns
    -------
    np.ndarray
        Epoched data with shape (n_channels, n_timepoints_per_epoch, n_epochs)
    
    Examples
    --------
    >>> data = np.random.randn(32, 10000)  # 40 seconds at 250 Hz
    >>> srate = 250.0
    >>> epochs = create_epochs(data, srate, epoch_duration=1.0)
    >>> print(epochs.shape)  # (32, 250, 40)
    """
    n_channels, n_timepoints = data.shape
    
    epoch_samples = int(epoch_duration * srate)
    step_samples = int((epoch_duration - overlap) * srate)
    
    # Calculate number of epochs
    n_epochs = (n_timepoints - epoch_samples) // step_samples + 1
    
    if n_epochs <= 0:
        raise ValueError(f"Data too short for epoch duration: {epoch_duration}s")
    
    # Create epochs
    epochs = np.zeros((n_channels, epoch_samples, n_epochs))
    
    for i in range(n_epochs):
        start = i * step_samples
        end = start + epoch_samples
        if end <= n_timepoints:
            epochs[:, :, i] = data[:, start:end]
    
    return epochs


def convert_to_short_epochs(
    data: np.ndarray,
    srate: float,
    epoch_duration: float = 1.0
) -> np.ndarray:
    """
    Convert epoched data back to continuous format.
    
    Parameters
    ----------
    data : np.ndarray
        Epoched data with shape (n_channels, n_timepoints_per_epoch, n_epochs)
    srate : float
        Sampling rate in Hz
    epoch_duration : float
        Duration of each epoch in seconds
    
    Returns
    -------
    np.ndarray
        Continuous data with shape (n_channels, n_timepoints)
    """
    if data.ndim == 3:
        n_channels, epoch_samples, n_epochs = data.shape
        return data.reshape(n_channels, -1)
    else:
        return data


def preprocess_for_microstates(
    eeg: EEGData,
    lowcut: float = 2.0,
    highcut: float = 30.0,
    use_avgref: bool = True,
    reject_artifacts: bool = True,
    art_threshold: float = 3.0,
    epoch_duration: float = 1.0
) -> EEGData:
    """
    Apply standard preprocessing pipeline for microstate analysis.
    
    This applies:
    1. Bandpass filtering (default 2-30 Hz)
    2. Average referencing (optional)
    3. Artifact rejection (optional)
    4. Epoch creation
    
    Parameters
    ----------
    eeg : EEGData
        Input EEG data
    lowcut : float
        Low cutoff frequency (default: 2.0 Hz)
    highcut : float
        High cutoff frequency (default: 30.0 Hz)
    use_avgref : bool
        Whether to apply average reference (default: True)
    reject_artifacts : bool
        Whether to reject artifacts (default: True)
    art_threshold : float
        Artifact rejection threshold in SD (default: 3.0)
    epoch_duration : float
        Epoch duration in seconds (default: 1.0)
    
    Returns
    -------
    EEGData
        Preprocessed EEG data
    
    Examples
    --------
    >>> from eeg_loader import generate_synthetic_eeg
    >>> eeg = generate_synthetic_eeg(n_channels=32, duration=30.0, srate=250.0)
    >>> eeg_preproc = preprocess_for_microstates(eeg)
    >>> print(eeg_preproc.data.shape)  # (32, n_timepoints)
    """
    if not eeg.has_data:
        raise ValueError("No data to preprocess")
    
    data = eeg.data.copy()
    
    # 1. Bandpass filter
    data = bandpass_filter(data, eeg.srate, lowcut=lowcut, highcut=highcut)
    
    # 2. Average reference
    if use_avgref:
        data, _ = average_reference(data)
    
    # 3. Artifact rejection
    if reject_artifacts:
        data = joint_probability_rejection(data, threshold=art_threshold)
    
    return EEGData(
        data=data,
        srate=eeg.srate,
        nbchan=eeg.nbchan,
        pnts=eeg.pnts,
        trials=eeg.trials,
        times=eeg.times,
        chan_info=eeg.chan_info,
        channel_labels=eeg.channel_labels,
        ref='average' if use_avgref else eeg.ref,
        events=eeg.events,
        event_types=eeg.event_types
    )


def downsample_data(
    data: np.ndarray,
    factor: int
) -> np.ndarray:
    """
    Downsample data by integer factor.
    
    Parameters
    ----------
    data : np.ndarray
        EEG data with shape (n_channels, n_timepoints)
    factor : int
        Downsampling factor
    
    Returns
    -------
    np.ndarray
        Downsampled data
    """
    if data.ndim == 1:
        return data[::factor]
    return data[:, ::factor]


def resample_data(
    data: np.ndarray,
    srate: float,
    new_srate: float
) -> np.ndarray:
    """
    Resample data to a new sampling rate.
    
    Uses scipy's resample function.
    
    Parameters
    ----------
    data : np.ndarray
        EEG data with shape (n_channels, n_timepoints)
    srate : float
        Current sampling rate
    new_srate : float
        Target sampling rate
    
    Returns
    -------
    np.ndarray
        Resampled data
    """
    ratio = new_srate / srate
    n_samples = int(data.shape[1] * ratio)
    
    if data.ndim == 1:
        return signal.resample(data, n_samples)
    
    # Resample each channel
    result = np.zeros((data.shape[0], n_samples))
    for ch in range(data.shape[0]):
        result[ch] = signal.resample(data[ch], n_samples)
    
    return result


def select_trials(
    data: np.ndarray,
    trial_indices: list
) -> np.ndarray:
    """
    Select specific trials from epoched data.
    
    Parameters
    ----------
    data : np.ndarray
        Epoched data with shape (n_channels, n_timepoints, n_trials)
    trial_indices : list
        List of trial indices to select
    
    Returns
    -------
    np.ndarray
        Selected trials
    """
    return data[:, :, trial_indices]


@dataclass
class PreprocessingResult:
    """Result of preprocessing operations."""
    data: np.ndarray
    original_srate: float
    new_srate: float
    filters_applied: list
    n_epochs: int
    epoch_duration: float
    rejected_samples: Optional[np.ndarray] = None


def preprocess_pipeline(
    eeg: EEGData,
    filter_band: tuple = (2.0, 30.0),
    resample_to: Optional[float] = None,
    apply_avgref: bool = True,
    artifact_threshold: Optional[float] = None,
    epoch_length: float = 1.0
) -> PreprocessingResult:
    """
    Complete preprocessing pipeline for microstate analysis.
    
    Parameters
    ----------
    eeg : EEGData
        Input EEG data
    filter_band : tuple
        (lowcut, highcut) filter band in Hz
    resample_to : float, optional
        Resample to this sampling rate if specified
    apply_avgref : bool
        Whether to apply average reference
    artifact_threshold : float, optional
        Artifact rejection threshold (None to skip)
    epoch_length : float
        Length of epochs in seconds
    
    Returns
    -------
    PreprocessingResult
        Preprocessed data and metadata
    """
    if not eeg.has_data:
        raise ValueError("No data to preprocess")
    
    data = eeg.data.copy()
    filters_applied = []
    current_srate = eeg.srate
    
    # 1. Filter
    data = bandpass_filter(data, current_srate, lowcut=filter_band[0], highcut=filter_band[1])
    filters_applied.append(f"bandpass_{filter_band[0]}_{filter_band[1]}")
    
    # 2. Resample if needed
    if resample_to is not None and resample_to != current_srate:
        data = resample_data(data, current_srate, resample_to)
        current_srate = resample_to
        filters_applied.append(f"resample_{resample_to}")
    
    # 3. Average reference
    if apply_avgref:
        data, _ = average_reference(data)
        filters_applied.append("average_reference")
    
    # 4. Artifact rejection
    rejected_samples = None
    if artifact_threshold is not None:
        data = joint_probability_rejection(data, threshold=artifact_threshold)
        # Mark rejected samples
        n_channels, n_timepoints = data.shape
        mean = np.mean(data, axis=1, keepdims=True)
        std = np.std(data, axis=1, keepdims=True)
        z_scores = np.abs((data - mean) / np.where(std == 0, 1, std))
        rejected_samples = np.any(z_scores > artifact_threshold, axis=0)
        filters_applied.append(f"artifact_rejection_{artifact_threshold}")
    
    # 5. Create epochs
    n_epochs = int(current_srate * epoch_length)
    epochs = data[:, :n_epochs * int(np.floor(data.shape[1] / n_epochs))]
    epochs = epochs.reshape(epochs.shape[0], n_epochs, -1)
    
    return PreprocessingResult(
        data=data,
        original_srate=eeg.srate,
        new_srate=current_srate,
        filters_applied=filters_applied,
        n_epochs=epochs.shape[2],
        epoch_duration=epoch_length,
        rejected_samples=rejected_samples
    )