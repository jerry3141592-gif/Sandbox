"""
EEG Microstate Analysis.

This module implements microstate segmentation analysis for EEG data,
including:
- Global Field Power (GFP) calculation
- GFP peak detection
- Modified k-means clustering for microstate prototype detection
- Microstate backfitting
- Microstate smoothing

These functions mirror EEGLAB's microstate analysis functions.
"""

import numpy as np
from scipy import signal
from scipy.cluster.hierarchy import fclusterdata
from scipy.spatial.distance import cdist
from typing import Tuple, List, Optional, Union
from dataclasses import dataclass, field
import warnings

from eeg_loader import EEGData

# Suppress runtime warnings for cleaner test output
warnings.filterwarnings('ignore', category=RuntimeWarning)


def compute_gfp(data: np.ndarray) -> np.ndarray:
    """
    Compute Global Field Power (GFP).
    
    GFP is the standard deviation of all channel voltages at each time point,
    providing a measure of the spatial complexity of the EEG at each moment.
    
    Parameters
    ----------
    data : np.ndarray
        EEG data with shape (n_channels, n_timepoints)
    
    Returns
    -------
    np.ndarray
        GFP time series with shape (n_timepoints,)
    
    Examples
    --------
    >>> data = np.random.randn(32, 5000)  # 32 channels, 5000 timepoints
    >>> gfp = compute_gfp(data)
    >>> print(gfp.shape)  # (5000,)
    """
    # GFP = standard deviation across channels at each time point
    gfp = np.std(data, axis=0)
    
    # Handle case where data might have NaN values
    if np.any(np.isnan(data)):
        gfp = np.nanstd(data, axis=0)
    
    return gfp


def compute_gfp_squared(data: np.ndarray) -> np.ndarray:
    """
    Compute squared GFP.
    
    The squared GFP is used in some microstate algorithms as it
    provides a better measure of signal strength.
    
    Parameters
    ----------
    data : np.ndarray
        EEG data with shape (n_channels, n_timepoints)
    
    Returns
    -------
    np.ndarray
        Squared GFP time series
    """
    gfp = compute_gfp(data)
    return gfp ** 2


def detect_gfp_peaks(
    gfp: np.ndarray,
    min_peak_distance: int = 10,
    threshold: float = 0.0,
    threshold_type: str = 'relative'
) -> np.ndarray:
    """
    Detect GFP peaks.
    
    GFP peaks indicate time points of maximum spatial stability and
    are used as the basis for microstate segmentation.
    
    Parameters
    ----------
    gfp : np.ndarray
        GFP time series
    min_peak_distance : int
        Minimum distance between peaks in samples (default: 10)
    threshold : float
        Threshold for peak detection
        - If 'absolute': absolute GFP value
        - If 'relative': fraction of max GFP (default: 0.0)
    threshold_type : str
        'absolute' or 'relative'
    
    Returns
    -------
    np.ndarray
        Indices of GFP peaks
    
    Examples
    --------
    >>> gfp = np.random.rand(5000)
    >>> peaks = detect_gfp_peaks(gfp, min_peak_distance=10)
    >>> print(len(peaks))  # Number of peaks detected
    """
    # Apply threshold
    if threshold_type == 'relative':
        threshold_val = threshold * np.max(gfp)
    else:
        threshold_val = threshold
    
    # Find local maxima
    peak_indices = signal.find_peaks(gfp, distance=min_peak_distance, height=threshold_val)[0]
    
    return np.array(peak_indices)


def extract_gfp_peaks_data(
    data: np.ndarray,
    gfp: np.ndarray,
    peak_indices: np.ndarray,
    window_size: int = 32
) -> np.ndarray:
    """
    Extract EEG data at GFP peaks.
    
    Extracts a temporal window around each GFP peak for clustering.
    
    Parameters
    ----------
    data : np.ndarray
        EEG data with shape (n_channels, n_timepoints)
    gfp : np.ndarray
        GFP time series
    peak_indices : np.ndarray
        Indices of GFP peaks
    window_size : int
        Window size around each peak (half on each side)
    
    Returns
    -------
    np.ndarray
        Extracted data with shape (n_peaks, n_channels * window_size)
    """
    n_channels = data.shape[0]
    n_peaks = len(peak_indices)
    
    # Prepare data for clustering
    peak_data = np.zeros((n_peaks, n_channels * window_size))
    
    for i, peak_idx in enumerate(peak_indices):
        # Extract window around peak
        start = max(0, peak_idx - window_size // 2)
        end = min(data.shape[1], peak_idx + window_size // 2)
        
        # Crop or pad as needed
        window_data = data[:, start:end]
        
        if window_data.shape[1] < window_size:
            # Pad with zeros if needed
            padded = np.zeros((n_channels, window_size))
            padded[:, :window_data.shape[1]] = window_data
            window_data = padded
        
        # Flatten
        peak_data[i] = window_data.flatten()
    
    return peak_data


@dataclass
class MicrostatePrototypes:
    """
    Microstate prototype data structure.
    
    Attributes
    ----------
    prototypes : np.ndarray
        Microstate prototype topographies with shape (n_microstates, n_channels)
    n_microstates : int
        Number of microstates
    n_channels : int
        Number of EEG channels
    explained_variance : float, optional
        Total explained variance
    map_corr : np.ndarray, optional
        Correlation between each map and prototype
    """
    prototypes: np.ndarray
    n_microstates: int
    n_channels: int
    explained_variance: Optional[float] = None
    map_corr: Optional[np.ndarray] = None
    sorting: str = 'global_explained_variance'
    
    def get_prototype(self, idx: int) -> np.ndarray:
        """Get a specific prototype."""
        return self.prototypes[idx]
    
    def sort_by_variance(self) -> 'MicrostatePrototypes':
        """Sort prototypes by explained variance."""
        if self.map_corr is not None:
            sorted_idx = np.argsort(self.map_corr)[::-1]
            return MicrostatePrototypes(
                prototypes=self.prototypes[sorted_idx],
                n_microstates=self.n_microstates,
                n_channels=self.n_channels,
                explained_variance=self.explained_variance,
                map_corr=self.map_corr[sorted_idx],
                sorting=self.sorting
            )
        return self


def modified_kmeans(
    data: np.ndarray,
    n_microstates: int,
    n_repetitions: int = 50,
    max_iterations: int = 1000,
    threshold: float = 1e-6,
    normalize: bool = False,
    random_seed: int = 42
) -> MicrostatePrototypes:
    """
    Modified k-means clustering for microstate detection.
    
    This implements the modified k-means algorithm used in EEGLAB's
    microstate segmentation (pop_micro_segment with algorithm='modkmeans').
    
    The algorithm uses GFP-weighted clustering where time points with
    higher GFP contribute more to the prototype estimation.
    
    Parameters
    ----------
    data : np.ndarray
        EEG data with shape (n_channels, n_timepoints)
    n_microstates : int
        Number of microstates to find
    n_repetitions : int
        Number of random initializations (default: 50)
    max_iterations : int
        Maximum iterations per run (default: 1000)
    threshold : float
        Convergence threshold (default: 1e-6)
    normalize : bool
        Whether to normalize data before clustering (default: False)
    random_seed : int
        Random seed for reproducibility
    
    Returns
    -------
    MicrostatePrototypes
        Microstate prototypes
    
    Examples
    --------
    >>> from eeg_loader import generate_synthetic_eeg
    >>> eeg = generate_synthetic_eeg(n_channels=32, duration=30.0, srate=250.0, n_microstates=4)
    >>> prototypes = modified_kmeans(eeg.data, n_microstates=4)
    >>> print(prototypes.n_microstates)  # 4
    """
    np.random.seed(random_seed)
    
    n_channels, n_timepoints = data.shape
    
    # Compute GFP for weighting
    gfp = compute_gfp(data)
    gfp_squared = gfp ** 2
    
    # Normalize data if requested
    if normalize:
        data_norm = data.copy()
        for t in range(n_timepoints):
            if gfp[t] > 0:
                data_norm[:, t] = data[:, t] / gfp[t]
    else:
        data_norm = data
    
    # Track best result across repetitions
    best_prototypes = None
    best_error = np.inf
    best_explained_variance = 0.0
    
    for rep in range(n_repetitions):
        # Initialize prototypes randomly from time points
        try:
            init_indices = np.random.choice(n_timepoints, n_microstates, replace=False)
            prototypes = data_norm[:, init_indices].T
        except ValueError:
            # Not enough unique time points - fallback to random initialization
            prototypes = np.random.randn(n_microstates, n_channels)
            for k in range(n_microstates):
                norm = np.linalg.norm(prototypes[k])
                if norm > 0:
                    prototypes[k] = prototypes[k] / norm
        
        # Normalize prototypes to unit norm
        for k in range(n_microstates):
            norm = np.linalg.norm(prototypes[k])
            if norm > 0:
                prototypes[k] = prototypes[k] / norm
        
        # Iterate
        prev_error = np.inf
        for iteration in range(max_iterations):
            # Assignment step: assign each time point to closest prototype
            # Use GFP-weighted correlation as distance
            distances = compute_correlation_distances(data_norm, prototypes, gfp)
            labels = np.argmin(distances, axis=1)
            
            # Update step: compute new prototypes
            new_prototypes = np.zeros_like(prototypes)
            for k in range(n_microstates):
                # Get time points assigned to this cluster
                mask = labels == k
                if np.sum(mask) > 0:
                    # GFP-weighted average
                    weights = gfp_squared[mask]
                    weights = weights / np.sum(weights)
                    new_prototypes[k] = np.average(data_norm[:, mask], axis=1, weights=weights)
                    
                    # Normalize
                    norm = np.linalg.norm(new_prototypes[k])
                    if norm > 0:
                        new_prototypes[k] = new_prototypes[k] / norm
            
            # Check convergence
            error = np.sum((prototypes - new_prototypes) ** 2)
            prototypes = new_prototypes
            
            if abs(prev_error - error) < threshold:
                break
            prev_error = error
        
        # Calculate explained variance
        explained_var = calculate_explained_variance(data, prototypes, labels, gfp)
        
        if explained_var > best_explained_variance:
            best_explained_variance = explained_var
            best_prototypes = prototypes.copy()
            best_labels = labels.copy()
    
    # Fallback if no valid prototypes found
    if best_prototypes is None:
        # Use the last prototypes as fallback
        best_prototypes = prototypes.copy()
        distances = compute_correlation_distances(data_norm, best_prototypes, gfp)
        best_labels = np.argmin(distances, axis=1)
        best_explained_variance = calculate_explained_variance(data, best_prototypes, best_labels, gfp)
    
    # Calculate map correlation
    map_corr = calculate_map_corr(data, best_prototypes)
    
    return MicrostatePrototypes(
        prototypes=best_prototypes,
        n_microstates=n_microstates,
        n_channels=n_channels,
        explained_variance=best_explained_variance,
        map_corr=map_corr,
        sorting='global_explained_variance'
    )


def compute_correlation_distances(
    data: np.ndarray,
    prototypes: np.ndarray,
    gfp: Optional[np.ndarray] = None
) -> np.ndarray:
    """
    Compute correlation-based distances between data and prototypes.
    
    Uses 1 - correlation as distance measure.
    
    Parameters
    ----------
    data : np.ndarray
        EEG data (n_channels, n_timepoints)
    prototypes : np.ndarray
        Prototype topographies (n_microstates, n_channels)
    gfp : np.ndarray, optional
        GFP values for weighting
    
    Returns
    -------
    np.ndarray
        Distances (n_timepoints, n_microstates)
    """
    n_timepoints = data.shape[1]
    n_microstates = prototypes.shape[0]
    
    # Normalize data and prototypes for efficient correlation computation
    # Using cosine similarity: corr = (data @ prototype) / (||data|| * ||prototype||)
    distances = np.zeros((n_timepoints, n_microstates))
    
    for k in range(n_microstates):
        prototype = prototypes[k]
        prototype_norm = np.linalg.norm(prototype)
        if prototype_norm == 0:
            prototype_norm = 1
        
        # Vectorized dot product computation
        dot_products = np.dot(data.T, prototype)  # (n_timepoints,)
        
        # Compute norm of each time point
        data_norms = np.linalg.norm(data, axis=0)  # (n_timepoints,)
        data_norms = np.where(data_norms == 0, 1, data_norms)
        
        # Correlation (cosine similarity)
        correlations = dot_products / (data_norms * prototype_norm)
        
        # Handle NaN values and take absolute value for polarity invariance
        correlations = np.where(np.isnan(correlations), 0, np.abs(correlations))
        
        # Distance is 1 - correlation
        distances[:, k] = 1 - correlations
    
    return distances


def calculate_explained_variance(
    data: np.ndarray,
    prototypes: np.ndarray,
    labels: np.ndarray,
    gfp: np.ndarray
) -> float:
    """
    Calculate explained variance.
    
    Parameters
    ----------
    data : np.ndarray
        EEG data
    prototypes : np.ndarray
        Microstate prototypes
    labels : np.ndarray
        Cluster labels for each time point
    gfp : np.ndarray
        GFP values
    
    Returns
    -------
    float
        Explained variance (0-1)
    """
    residual_var = calculate_residual_variance(data, prototypes, labels, gfp)
    total_var = np.sum(gfp ** 2)
    
    if total_var == 0:
        return 0.0
    
    # Explained variance = 1 - residual variance / total variance
    explained = 1.0 - (residual_var / total_var)
    
    return max(0.0, min(1.0, explained))


def calculate_residual_variance(
    data: np.ndarray,
    prototypes: np.ndarray,
    labels: np.ndarray,
    gfp: np.ndarray
) -> float:
    """Calculate residual variance."""
    n_microstates = prototypes.shape[0]
    residual = 0.0
    
    for k in range(n_microstates):
        mask = labels == k
        if np.sum(mask) > 0:
            # Get data for this cluster
            cluster_data = data[:, mask]  # shape: (n_channels, n_cluster_timepoints)
            gfp_mask = gfp[mask]  # shape: (n_cluster_timepoints,)
            prototype = prototypes[k]  # shape: (n_channels,)
            
            # Compute difference: data[:, t] - prototype (for each time point)
            # diff has shape (n_channels, n_cluster_timepoints)
            diff = cluster_data - prototype[:, np.newaxis]
            
            # Weight by GFP squared and sum
            # gfp_mask[:, np.newaxis] has shape (n_cluster_timepoints, 1)
            # diff ** 2 has shape (n_channels, n_cluster_timepoints)
            weighted_diff = gfp_mask[np.newaxis, :] ** 2 * diff ** 2
            residual += np.sum(weighted_diff)
    
    return residual


def calculate_map_corr(
    data: np.ndarray,
    prototypes: np.ndarray
) -> np.ndarray:
    """
    Calculate correlation between best matching prototype and data at each time point.
    
    Parameters
    ----------
    data : np.ndarray
        EEG data (n_channels, n_timepoints)
    prototypes : np.ndarray
        Prototypes (n_microstates, n_channels)
    
    Returns
    -------
    np.ndarray
        Mean correlation for each microstate
    """
    n_microstates, n_channels = prototypes.shape
    n_timepoints = data.shape[1]
    
    # Best matching prototype for each time point
    distances = compute_correlation_distances(data, prototypes)
    bmp_labels = np.argmin(distances, axis=1)
    
    # Calculate correlation for each microstate
    map_corr = np.zeros(n_microstates)
    for k in range(n_microstates):
        mask = bmp_labels == k
        if np.sum(mask) > 0:
            correlations = []
            for t in np.where(mask)[0]:
                corr = np.abs(np.corrcoef(data[:, t], prototypes[k])[0, 1])
                if not np.isnan(corr):
                    correlations.append(corr)
            if correlations:
                map_corr[k] = np.mean(correlations)
    
    return map_corr


def find_optimal_nmicrostates(
    data: np.ndarray,
    n_range: range = range(2, 9),
    n_repetitions: int = 20,
    criterion: str = 'global_explained_variance'
) -> Tuple[List[MicrostatePrototypes], np.ndarray]:
    """
    Find optimal number of microstates.
    
    Tests different numbers of microstates and selects based on
    the chosen criterion (e.g., explained variance, cross-validation).
    
    Parameters
    ----------
    data : np.ndarray
        EEG data
    n_range : range
        Range of number of microstates to test (e.g., range(2, 9))
    n_repetitions : int
        Number of repetitions for each n_microstates
    criterion : str
        Selection criterion: 'global_explained_variance' or 'cv'
    
    Returns
    -------
    Tuple[List[MicrostatePrototypes], np.ndarray]
        - List of results for each n_microstates
        - Criterion values for each n_microstates
    """
    results = []
    criterion_values = []
    
    for n_microstates in n_range:
        result = modified_kmeans(data, n_microstates, n_repetitions=n_repetitions)
        results.append(result)
        
        if criterion == 'global_explained_variance':
            criterion_values.append(result.explained_variance)
        elif criterion == 'cv':
            # Cross-validation error
            criterion_values.append(calculate_cv_error(data, result.prototypes))
    
    return results, np.array(criterion_values)


def calculate_cv_error(
    data: np.ndarray,
    prototypes: np.ndarray
) -> float:
    """
    Calculate cross-validation error.
    
    Lower is better.
    
    Parameters
    ----------
    data : np.ndarray
        EEG data
    prototypes : np.ndarray
        Microstate prototypes
    
    Returns
    -------
    float
        Cross-validation error
    """
    gfp = compute_gfp(data)
    distances = compute_correlation_distances(data, prototypes, gfp)
    
    # Minimize sum of distances
    cv_error = np.sum(np.min(distances, axis=1))
    
    return cv_error


@dataclass
class MicrostateSequence:
    """
    Microstate sequence data structure.
    
    Attributes
    ----------
    labels : np.ndarray
        Microstate label for each time point
    timing : np.ndarray
        Timing information for each microstate
    n_microstates : int
        Number of microstates
    prototypes : MicrostatePrototypes
        The microstate prototypes
    """
    labels: np.ndarray
    n_microstates: int
    prototypes: MicrostatePrototypes
    smoothing: Optional[str] = None
    
    @property
    def n_timepoints(self) -> int:
        """Number of time points."""
        return len(self.labels)
    
    def get_segment_durations(self) -> np.ndarray:
        """Get duration of each microstate segment."""
        durations = []
        if len(self.labels) == 0:
            return np.array(durations)
        
        current_label = self.labels[0]
        current_duration = 1
        
        for i in range(1, len(self.labels)):
            if self.labels[i] == current_label:
                current_duration += 1
            else:
                durations.append(current_duration)
                current_label = self.labels[i]
                current_duration = 1
        
        # Add last segment
        durations.append(current_duration)
        
        return np.array(durations)
    
    def get_microstate_counts(self) -> np.ndarray:
        """Get count of each microstate."""
        counts = np.zeros(self.n_microstates)
        for label in self.labels:
            if label >= 0:
                counts[label] += 1
        return counts
    
    def get_microstate_ratios(self) -> np.ndarray:
        """Get ratio (coverage) of each microstate."""
        counts = self.get_microstate_counts()
        return counts / len(self.labels) if len(self.labels) > 0 else counts


def backfit_microstates(
    data: np.ndarray,
    prototypes: MicrostatePrototypes,
    polarity_invariance: bool = True
) -> MicrostateSequence:
    """
    Backfit microstate prototypes to full EEG data.
    
    Assigns each time point to the best matching microstate prototype.
    
    Parameters
    ----------
    data : np.ndarray
        EEG data (n_channels, n_timepoints)
    prototypes : MicrostatePrototypes
        Microstate prototypes
    polarity_invariance : bool
        Whether to use polarity invariance (default: True)
    
    Returns
    -------
    MicrostateSequence
        Microstate labels for each time point
    
    Examples
    --------
    >>> from eeg_loader import generate_synthetic_eeg
    >>> eeg = generate_synthetic_eeg(n_channels=32, duration=30.0, srate=250.0, n_microstates=4)
    >>> prototypes = modified_kmeans(eeg.data, n_microstates=4)
    >>> seq = backfit_microstates(eeg.data, prototypes)
    >>> print(seq.labels.shape)  # (n_timepoints,)
    """
    # Compute distances
    gfp = compute_gfp(data)
    distances = compute_correlation_distances(data, prototypes.prototypes, gfp)
    
    # Assign to best matching prototype
    labels = np.argmin(distances, axis=1)
    
    return MicrostateSequence(
        labels=labels,
        n_microstates=prototypes.n_microstates,
        prototypes=prototypes,
        smoothing=None
    )


def smooth_microstate_labels(
    labels: np.ndarray,
    min_duration: int = 30,
    smoothing_type: str = 'reject'
) -> np.ndarray:
    """
    Smooth microstate labels by removing short segments.
    
    Parameters
    ----------
    labels : np.ndarray
        Microstate labels
    min_duration : int
        Minimum duration for a segment (in samples)
    smoothing_type : str
        'reject' or 'annotate'
    
    Returns
    -------
    np.ndarray
        Smoothed labels
    
    Examples
    --------
    >>> labels = np.array([0, 0, 0, 1, 1, 2, 2, 0, 0])
    >>> smoothed = smooth_microstate_labels(labels, min_duration=3)
    >>> print(smoothed)  # Labels with short segments removed
    """
    n = len(labels)
    result = labels.copy()
    
    # Find segments and their durations
    segments = []
    current_label = labels[0]
    start = 0
    
    for i in range(1, n):
        if labels[i] != current_label:
            duration = i - start
            segments.append((start, i - 1, current_label, duration))
            current_label = labels[i]
            start = i
    
    # Add last segment
    segments.append((start, n - 1, current_label, n - start))
    
    # Process short segments
    if smoothing_type == 'reject':
        for start, end, label, duration in segments:
            if duration < min_duration:
                result[start:end+1] = -1  # Mark as rejected
    elif smoothing_type == 'annotate':
        # Replace short segments with neighboring dominant state
        for start, end, label, duration in segments:
            if duration < min_duration:
                # Get dominant neighbor
                if start > 0 and end < n - 1:
                    # Either before or after
                    before_duration = segments[segments.index((start, start-1 if start > 0 else 0, labels[start-1] if start > 0 else 0, 0))][3] if start > 0 else 0
                    after_duration = segments[segments.index((end+1, end+1, labels[end+1] if end+1 < n else 0, 0))][3] if end + 1 < n else 0
                    
                    if before_duration > after_duration and start > 0:
                        result[start:end+1] = labels[start-1]
                    elif end + 1 < n:
                        result[start:end+1] = labels[end+1]
                elif start > 0:
                    result[start:end+1] = labels[start-1]
                elif end + 1 < n:
                    result[start:end+1] = labels[end+1]
    
    return result


def calculate_microstate_statistics(
    sequence: MicrostateSequence,
    srate: float
) -> dict:
    """
    Calculate microstate statistics.
    
    Parameters
    ----------
    sequence : MicrostateSequence
        Microstate sequence
    srate : float
        Sampling rate in Hz
    
    Returns
    -------
    dict
        Dictionary of statistics
    """
    # Get segment durations in ms
    durations_ms = sequence.get_segment_durations() / srate * 1000
    
    # Get microstate coverage
    coverage = sequence.get_microstate_ratios()
    
    # Get microstate counts
    counts = sequence.get_microstate_counts()
    
    # Mean and SD of durations for each microstate
    mean_durations = {}
    sd_durations = {}
    
    for k in range(sequence.n_microstates):
        mask = sequence.labels == k
        if np.sum(mask) > 0:
            # Find segments for this microstate
            try:
                # Create boundary detection array
                left = np.array([False])
                middle = (mask[:-1] != mask[1:]).astype(bool)
                right = np.array([False])
                boundaries = np.concatenate([left, middle, right])
                boundary_indices = np.where(boundaries)[0]
                if len(boundary_indices) > 1:
                    segments = np.diff(boundary_indices)
                    mean_durations[k] = np.mean(segments) / srate * 1000
                    sd_durations[k] = np.std(segments) / srate * 1000
            except Exception as e:
                mean_durations[k] = 0.0
                sd_durations[k] = 0.0
    
    return {
        'coverage': coverage,
        'counts': counts,
        'mean_duration_ms': mean_durations,
        'sd_duration_ms': sd_durations,
        'total_time_ms': np.sum(durations_ms),
        'n_segments': len(durations_ms)
    }


def segment_microstates(
    eeg: EEGData,
    n_microstates: int = 4,
    algorithm: str = 'modkmeans',
    sorting: str = 'global_explained_variance',
    n_repetitions: int = 50,
    max_iterations: int = 1000,
    threshold: float = 1e-6,
    smoothing: str = 'none',
    smoothing_min_duration: int = 30
) -> Tuple[MicrostatePrototypes, MicrostateSequence]:
    """
    Complete microstate segmentation pipeline.
    
    This is the main function for microstate analysis, implementing
    the full pipeline from EEGLAB's microstate analysis.
    
    Parameters
    ----------
    eeg : EEGData
        Input EEG data
    n_microstates : int
        Number of microstates to find (default: 4)
    algorithm : str
        Clustering algorithm (default: 'modkmeans')
    sorting : str
        How to sort microstates (default: 'global_explained_variance')
    n_repetitions : int
        Number of random initializations (default: 50)
    max_iterations : int
        Maximum iterations (default: 1000)
    threshold : float
        Convergence threshold (default: 1e-6)
    smoothing : str
        Smoothing type: 'none', 'reject', or 'annotate'
    smoothing_min_duration : int
        Minimum segment duration for smoothing
    
    Returns
    -------
    Tuple[MicrostatePrototypes, MicrostateSequence]
        - Microstate prototypes
        - Microstate sequence
    
    Examples
    --------
    >>> from eeg_loader import generate_synthetic_eeg
    >>> eeg = generate_synthetic_eeg(n_channels=32, duration=30.0, srate=250.0, n_microstates=4)
    >>> prototypes, sequence = segment_microstates(eeg, n_microstates=4)
    >>> print(f'Found {prototypes.n_microstates} microstates')
    """
    if not eeg.has_data:
        raise ValueError("No EEG data available")
    
    # Step 1: Cluster to find prototypes
    prototypes = modified_kmeans(
        eeg.data,
        n_microstates=n_microstates,
        n_repetitions=n_repetitions,
        max_iterations=max_iterations,
        threshold=threshold
    )
    
    # Step 2: Backfit to full data
    sequence = backfit_microstates(eeg.data, prototypes)
    
    # Step 3: Apply smoothing if requested
    if smoothing != 'none':
        smoothed_labels = smooth_microstate_labels(
            sequence.labels,
            min_duration=smoothing_min_duration,
            smoothing_type=smoothing
        )
        sequence.labels = smoothed_labels
        sequence.smoothing = smoothing
    
    return prototypes, sequence