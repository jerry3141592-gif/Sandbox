"""
EEG Loader for EEGLAB .set format files.

This module provides functionality to load EEG data from EEGLAB .set files,
handling both data embedded in the .set file and data referenced via .fdt files.

Supports loading via MNE-Python when .fdt files are available, or returns
metadata only when data is not available.
"""

import os
import numpy as np
import scipy.io as sio
from dataclasses import dataclass
from typing import Optional, List, Tuple, Union, Any

# Try to import MNE for better .set file support
try:
    import mne
    MNE_AVAILABLE = True
except ImportError:
    MNE_AVAILABLE = False


@dataclass
class ChannelInfo:
    """Channel location information."""
    labels: np.ndarray
    types: np.ndarray
    theta: np.ndarray
    radius: np.ndarray
    X: np.ndarray
    Y: np.ndarray
    Z: np.ndarray
    sph_theta: np.ndarray
    sph_phi: np.ndarray
    sph_radius: np.ndarray


@dataclass
class EEGData:
    """
    EEG data structure.
    
    Attributes
    ----------
    data : np.ndarray
        EEG data with shape (n_channels, n_timepoints) or (n_channels, n_timepoints, n_trials)
    srate : float
        Sampling rate in Hz
    nbchan : int
        Number of channels
    pnts : int
        Number of time points
    trials : int
        Number of trials (epochs)
    times : np.ndarray, optional
        Time axis values
    chan_info : ChannelInfo, optional
        Channel location information
    channel_labels : List[str], optional
        List of channel labels
    ref : str, optional
        Reference electrode type
    events : np.ndarray, optional
        Event latencies
    event_types : np.ndarray, optional
        Event types
    """
    data: Optional[np.ndarray]  # Shape: (n_channels, n_timepoints)
    srate: float
    nbchan: int
    pnts: int
    trials: int
    times: Optional[np.ndarray] = None
    chan_info: Optional[ChannelInfo] = None
    channel_labels: Optional[List[str]] = None
    ref: Optional[str] = None
    events: Optional[np.ndarray] = None
    event_types: Optional[np.ndarray] = None
    file_path: Optional[str] = None
    
    def __post_init__(self):
        """Validate data after initialization."""
        if self.times is None and self.srate > 0:
            self.times = np.arange(self.pnts) / self.srate
    
    @property
    def n_channels(self) -> int:
        """Number of channels."""
        return self.nbchan
    
    @property
    def n_timepoints(self) -> int:
        """Number of time points."""
        return self.pnts
    
    @property
    def duration(self) -> float:
        """Recording duration in seconds."""
        return self.pnts / self.srate if self.srate > 0 else 0
    
    @property
    def has_data(self) -> bool:
        """Check if actual data is loaded."""
        return self.data is not None and self.data.size > 0


def load_set_file(filepath: str, load_data: bool = True) -> EEGData:
    """
    Load an EEGLAB .set file.
    
    Tries multiple loading strategies:
    1. MNE-Python (best support for .set + .fdt pairs)
    2. Direct .fdt loading as fallback
    
    Parameters
    ----------
    filepath : str
        Path to the .set file
    load_data : bool
        If True, attempt to load the data. If False, only return metadata.
    
    Returns
    -------
    EEGData
        EEG data structure with metadata and optionally data
    
    Examples
    --------
    >>> eeg = load_set_file('data.set')
    >>> print(eeg.n_channels, eeg.n_timepoints)
    """
    # Load metadata from .set file
    mat = sio.loadmat(filepath, squeeze_me=True, struct_as_record=False)
    
    # Get basic info
    nbchan = int(mat['nbchan'])
    pnts = int(mat['pnts'])
    trials = int(mat['trials'])
    srate = float(mat['srate'])
    
    # Get reference
    ref = str(mat.get('ref', 'unknown'))
    
    # Get channel labels
    channel_labels = _extract_channel_labels(mat.get('chanlocs', None))
    
    # Get times
    times = None
    if 'times' in mat:
        times_arr = mat['times']
        if times_arr is not None and len(times_arr) > 0:
            if isinstance(times_arr, np.ndarray):
                times = times_arr.flatten()
            else:
                times = np.array(list(times_arr))
    
    # Get events
    events, event_types = _extract_events(mat.get('event', None))
    
    # Load data
    data = None
    if load_data:
        # Try MNE first (best for handling .set with .fdt reference)
        data = _load_using_mne(filepath)
        
        # If MNE failed, try direct .fdt loading
        if data is None:
            data = _load_fdt_directly(filepath, nbchan, pnts, trials)
    
    return EEGData(
        data=data,
        srate=srate,
        nbchan=nbchan,
        pnts=pnts,
        trials=trials,
        times=times,
        chan_info=None,
        channel_labels=channel_labels,
        ref=ref,
        events=events,
        event_types=event_types,
        file_path=filepath
    )


def _extract_channel_labels(chanlocs: Any) -> Optional[List[str]]:
    """Extract channel labels from chanlocs structure."""
    if chanlocs is None:
        return None
    
    try:
        # Handle different chanlocs formats
        if hasattr(chanlocs, '__len__'):
            if len(chanlocs) == 0:
                return None
            
            # Try to extract labels
            if hasattr(chanlocs[0], 'labels'):
                return [str(ch.labels) for ch in chanlocs]
            elif isinstance(chanlocs, dict) and 'labels' in chanlocs:
                labels_arr = chanlocs['labels']
                if hasattr(labels_arr, '__iter__'):
                    return list(labels_arr)
        return None
    except:
        return None


def _extract_events(event_struct: Any) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    """Extract events from event structure."""
    if event_struct is None:
        return None, None
    
    try:
        if hasattr(event_struct, '__len__'):
            n_events = len(event_struct) if len(event_struct) > 0 else 0
        else:
            n_events = 1
        
        events = np.zeros(n_events)
        event_types = []
        
        for i, ev in enumerate(event_struct):
            if hasattr(ev, 'latency'):
                events[i] = ev.latency
            if hasattr(ev, 'type'):
                event_types.append(str(ev.type))
        
        if event_types:
            return events, np.array(event_types)
        return None, None
    except:
        return None, None


def _load_using_mne(filepath: str) -> Optional[np.ndarray]:
    """Try loading using MNE-Python."""
    if not MNE_AVAILABLE:
        return None
    
    try:
        # Try MNE loader
        base_dir = os.path.dirname(filepath)
        set_filename = os.path.basename(filepath)
        
        # First, check if .fdt file exists in same directory
        mat = sio.loadmat(filepath, squeeze_me=True, struct_as_record=False)
        data_ref = mat.get('data', '')
        
        if data_ref:
            fdt_filename = data_ref if isinstance(data_ref, str) else data_ref[0] if len(data_ref) > 0 else None
            if fdt_filename:
                fdt_path = os.path.join(base_dir, fdt_filename)
                if os.path.exists(fdt_path):
                    # Use MNE to load
                    raw = mne.io.read_raw_eeglab(filepath, preload=True, verbose=False)
                    return raw.get_data()
        
        return None
    except Exception as e:
        return None


def _load_fdt_directly(filepath: str, nbchan: int, pnts: int, trials: int) -> Optional[np.ndarray]:
    """Try loading .fdt file directly."""
    try:
        base_dir = os.path.dirname(filepath)
        mat = sio.loadmat(filepath, squeeze_me=True, struct_as_record=False)
        data_ref = mat.get('data', '')
        
        if data_ref:
            fdt_filename = data_ref if isinstance(data_ref, str) else data_ref[0] if len(data_ref) > 0 else None
            if fdt_filename:
                fdt_path = os.path.join(base_dir, fdt_filename)
                if os.path.exists(fdt_path):
                    # Load binary data (float64, MATLAB column-major)
                    data = np.fromfile(fdt_path, dtype=np.float64)
                    
                    # Reshape - column major (MATLAB style)
                    if trials > 1:
                        data = data.reshape(nbchan, pnts, trials, order='F')
                    else:
                        data = data.reshape(nbchan, pnts, order='F')
                    
                    return data
        return None
    except:
        return None


def load_set_files_from_directory(directory: str, pattern: str = "*.set") -> List[EEGData]:
    """
    Load all .set files from a directory matching the pattern.
    
    Parameters
    ----------
    directory : str
        Directory containing .set files
    pattern : str
        Glob pattern for matching files (default: "*.set")
    
    Returns
    -------
    List[EEGData]
        List of loaded EEG data structures
    """
    import glob
    
    # Get list of .set files
    search_pattern = os.path.join(directory, pattern)
    set_files = glob.glob(search_pattern)
    
    # Sort files
    set_files = sorted(set_files)
    
    # Load each file
    eeg_list = []
    for filepath in set_files:
        try:
            eeg = load_set_file(filepath, load_data=True)
            eeg_list.append(eeg)
        except Exception as e:
            print(f"Warning: Failed to load {filepath}: {e}")
    
    return eeg_list


def concatenate_eeg_data(eeg_list: List[EEGData], axis: int = 1) -> EEGData:
    """
    Concatenate multiple EEG data structures along the time axis.
    
    Parameters
    ----------
    eeg_list : List[EEGData]
        List of EEG data structures to concatenate
    axis : int
        Axis along which to concatenate (1 = time, 0 = channels)
    
    Returns
    -------
    EEGData
        Concatenated EEG data
    """
    if not eeg_list:
        raise ValueError("Empty list provided")
    
    # Check that all EEG data have same number of channels and sampling rate
    nbchan = eeg_list[0].nbchan
    srate = eeg_list[0].srate
    
    for eeg in eeg_list:
        if eeg.nbchan != nbchan:
            raise ValueError(f"Channel mismatch: {eeg.nbchan} != {nbchan}")
        if eeg.srate != srate:
            raise ValueError(f"Sampling rate mismatch: {eeg.srate} != {srate}")
    
    # Concatenate data along time axis
    data_list = [eeg.data for eeg in eeg_list if eeg.has_data]
    if data_list:
        data = np.concatenate(data_list, axis=1)
    else:
        data = None
    
    # Calculate total time points
    pnts = sum(eeg.pnts for eeg in eeg_list)
    
    # Concatenate times
    times_list = [eeg.times for eeg in eeg_list if eeg.times is not None]
    if times_list:
        times = np.concatenate(times_list)
    else:
        times = None
    
    return EEGData(
        data=data,
        srate=srate,
        nbchan=nbchan,
        pnts=pnts,
        trials=1,
        times=times,
        chan_info=eeg_list[0].chan_info,
        channel_labels=eeg_list[0].channel_labels,
        ref=eeg_list[0].ref,
        events=None,
        event_types=None
    )


def generate_synthetic_eeg(
    n_channels: int = 35,
    duration: float = 60.0,
    srate: float = 500.0,
    n_microstates: int = 4,
    noise_level: float = 0.5,
    seed: int = 42
) -> EEGData:
    """
    Generate synthetic EEG data for testing microstate algorithms.
    
    This creates realistic EEG-like data with distinct microstate patterns
    that can be used to test microstate segmentation algorithms.
    
    Parameters
    ----------
    n_channels : int
        Number of EEG channels
    duration : float
        Recording duration in seconds
    srate : float
        Sampling rate in Hz
    n_microstates : int
        Number of distinct microstate prototypes to generate
    noise_level : float
        Level of random noise to add (relative to signal)
    seed : int
        Random seed for reproducibility
    
    Returns
    -------
    EEGData
        Synthetic EEG data with shape (n_channels, n_timepoints)
    
    Examples
    --------
    >>> eeg = generate_synthetic_eeg(n_channels=32, duration=30.0, srate=250.0)
    >>> print(eeg.data.shape)  # (32, 7500)
    """
    np.random.seed(seed)
    
    n_timepoints = int(duration * srate)
    
    # Generate random microstate prototype topographies
    # Each prototype is a spatial pattern across channels
    prototypes = np.random.randn(n_microstates, n_channels)
    
    # Normalize prototypes to unit variance
    for i in range(n_microstates):
        prototypes[i] = prototypes[i] / np.std(prototypes[i])
    
    # Generate microstate sequence (which microstate at each time point)
    # Use a hidden Markov model-like approach for realistic transitions
    min_duration = int(0.02 * srate)  # Minimum 20ms duration
    max_duration = int(0.5 * srate)    # Maximum 500ms duration
    
    state_sequence = np.zeros(n_timepoints, dtype=int)
    t = 0
    while t < n_timepoints:
        # Pick a random microstate
        state = np.random.randint(0, n_microstates)
        # Determine how long to stay in this state
        duration = np.random.randint(min_duration, max_duration)
        # Assign state
        end_t = min(t + duration, n_timepoints)
        state_sequence[t:end_t] = state
        t = end_t
    
    # Generate EEG data from prototypes and state sequence
    data = np.zeros((n_channels, n_timepoints))
    for t in range(n_timepoints):
        state = state_sequence[t]
        # Add microstate contribution + noise
        noise = np.random.randn(n_channels) * noise_level
        data[:, t] = prototypes[state] + noise
    
    # Create time axis
    times = np.arange(n_timepoints) / srate
    
    # Generate default channel labels
    channel_labels = [f'EEG{i+1}' for i in range(n_channels)]
    
    return EEGData(
        data=data,
        srate=srate,
        nbchan=n_channels,
        pnts=n_timepoints,
        trials=1,
        times=times,
        chan_info=None,
        channel_labels=channel_labels,
        ref='average',
        events=None,
        event_types=None
    )


def load_metadata_from_set(filepath: str) -> EEGData:
    """
    Load only metadata from an EEGLAB .set file (without data).
    
    Useful when you only need information about the file without
    loading the potentially large data arrays.
    
    Parameters
    ----------
    filepath : str
        Path to the .set file
    
    Returns
    -------
    EEGData
        EEG data structure with metadata only (data will be None)
    """
    return load_set_file(filepath, load_data=False)