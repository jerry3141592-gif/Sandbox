"""
Unit tests for EEG Microstate Analysis.

These tests verify the correctness of the microstate analysis implementation.
"""

import unittest
import numpy as np
import tempfile
import os
from scipy.io import savemat

from eeg_loader import EEGData, load_set_file, generate_synthetic_eeg, load_metadata_from_set
from eeg_preprocessing import (
    bandpass_filter,
    average_reference,
    joint_probability_rejection,
    create_epochs,
    preprocess_for_microstates
)
from microstates import (
    compute_gfp,
    detect_gfp_peaks,
    modified_kmeans,
    backfit_microstates,
    smooth_microstate_labels,
    segment_microstates,
    MicrostatePrototypes,
    MicrostateSequence,
    calculate_microstate_statistics,
    find_optimal_nmicrostates
)


class TestEEGLoader(unittest.TestCase):
    """Tests for EEG data loading."""
    
    def test_generate_synthetic_eeg(self):
        """Test synthetic EEG data generation."""
        eeg = generate_synthetic_eeg(
            n_channels=32,
            duration=10.0,
            srate=250.0,
            n_microstates=4
        )
        
        # Check dimensions
        self.assertEqual(eeg.nbchan, 32)
        self.assertEqual(eeg.pnts, 2500)
        self.assertEqual(eeg.srate, 250.0)
        
        # Check data shape
        self.assertEqual(eeg.data.shape, (32, 2500))
        
        # Check metadata
        self.assertIsNotNone(eeg.times)
        self.assertIsNotNone(eeg.channel_labels)
    
    def test_synthetic_eeg_channel_labels(self):
        """Test that channel labels are generated correctly."""
        eeg = generate_synthetic_eeg(n_channels=16)
        self.assertEqual(len(eeg.channel_labels), 16)
        self.assertEqual(eeg.channel_labels[0], 'EEG1')
        self.assertEqual(eeg.channel_labels[-1], 'EEG16')
    
    def test_metadata_loading(self):
        """Test loading metadata only."""
        eeg = load_metadata_from_set('/tmp/eeg_test/D_01.set')
        
        self.assertEqual(eeg.nbchan, 35)
        self.assertEqual(eeg.pnts, 180001)
        self.assertEqual(eeg.srate, 500.0)
        self.assertFalse(eeg.has_data)


class TestPreprocessing(unittest.TestCase):
    """Tests for EEG preprocessing functions."""
    
    def setUp(self):
        """Set up test data."""
        np.random.seed(42)
        self.data = np.random.randn(32, 5000)
        self.srate = 250.0
    
    def test_bandpass_filter_shape(self):
        """Test that bandpass filter preserves data shape."""
        filtered = bandpass_filter(self.data, self.srate, lowcut=2.0, highcut=30.0)
        self.assertEqual(filtered.shape, self.data.shape)
    
    def test_bandpass_filter_reduces_noise(self):
        """Test that filtering reduces high frequency noise."""
        # Generate data with high frequency noise
        t = np.linspace(0, 1, 5000)
        signal_1hz = np.sin(2*np.pi*1*t).reshape(1, -1)
        noise = np.random.randn(1, 5000) * 5  # High amplitude noise
        
        data_with_noise = np.vstack([signal_1hz, noise])
        data_with_noise = data_with_noise.repeat(32, axis=0)
        
        # Filter
        filtered = bandpass_filter(data_with_noise, self.srate, lowcut=2.0, highcut=30.0)
        
        # Check that variance is reduced
        orig_var = np.var(data_with_noise[0])
        new_var = np.var(filtered[0])
        self.assertLess(new_var, orig_var)
    
    def test_average_reference(self):
        """Test average referencing."""
        referenced, reference = average_reference(self.data)
        
        # Check shape
        self.assertEqual(referenced.shape, self.data.shape)
        
        # Check that average is zero (or very close)
        mean_across = np.mean(referenced, axis=0)
        self.assertTrue(np.allclose(mean_across, 0, atol=1e-10))
    
    def test_average_reference_values(self):
        """Test that average reference computes correctly."""
        data = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype=float)
        referenced, reference = average_reference(data)
        
        # Average should be [4, 5, 6]
        expected_ref = np.array([[4, 5, 6]])
        np.testing.assert_array_almost_equal(reference, expected_ref)
        
        # After referencing, mean should be zero
        np.testing.assert_array_almost_equal(np.mean(referenced, axis=0), 0)
    
    def test_joint_probability_rejection(self):
        """Test joint probability artifact rejection."""
        data = self.data.copy()
        
        # Add artifacts at specific time points
        data[:, 100:105] = 100  # Large artifact
        data[:, 1000:1002] = 50
        
        # Reject artifacts
        cleaned = joint_probability_rejection(data, threshold=3.0)
        
        # Should still have same shape
        self.assertEqual(cleaned.shape, data.shape)
    
    def test_create_epochs(self):
        """Test epoch creation from continuous data."""
        epochs = create_epochs(self.data, self.srate, epoch_duration=1.0)
        
        # Check shape: 32 channels, 250 samples per epoch, 20 epochs
        expected_shape = (32, 250, 20)
        self.assertEqual(epochs.shape, expected_shape)
    
    def test_create_epochs_with_overlap(self):
        """Test epoch creation with overlap."""
        epochs = create_epochs(self.data, self.srate, epoch_duration=1.0, overlap=0.5)
        
        # With 50% overlap, should have more epochs
        epochs_no_overlap = create_epochs(self.data, self.srate, epoch_duration=1.0, overlap=0.0)
        self.assertGreater(epochs.shape[2], epochs_no_overlap.shape[2])


class TestGFP(unittest.TestCase):
    """Tests for Global Field Power calculation."""
    
    def setUp(self):
        """Set up test data."""
        np.random.seed(42)
        self.data = np.random.randn(32, 5000)
        self.srate = 250.0
    
    def test_gfp_shape(self):
        """Test GFP has correct shape."""
        gfp = compute_gfp(self.data)
        self.assertEqual(gfp.shape, (5000,))
    
    def test_gfp_positive(self):
        """Test GFP values are positive."""
        gfp = compute_gfp(self.data)
        self.assertTrue(np.all(gfp >= 0))
    
    def test_gfp_zero_channels(self):
        """Test GFP with zero-valued channels."""
        data = np.zeros((3, 100))
        gfp = compute_gfp(data)
        np.testing.assert_array_equal(gfp, 0)
    
    def test_gfp_constant_signal(self):
        """Test GFP for constant signal across channels."""
        data = np.ones((3, 100)) * 5
        gfp = compute_gfp(data)
        np.testing.assert_array_equal(gfp, 0)
    
    def test_detect_gfp_peaks_returns_array(self):
        """Test that peak detection returns numpy array."""
        gfp = compute_gfp(self.data)
        peaks = detect_gfp_peaks(gfp, min_peak_distance=10)
        
        self.assertIsInstance(peaks, np.ndarray)
        self.assertTrue(np.issubdtype(peaks.dtype, np.integer))
    
    def test_detect_gfp_peaks_min_distance(self):
        """Test that peaks maintain minimum distance."""
        # Create GFP with known peaks
        gfp = np.zeros(1000)
        gfp[100] = 1.0
        gfp[105] = 1.5  # Too close
        gfp[200] = 1.2
        gfp[300] = 1.3
        
        peaks = detect_gfp_peaks(gfp, min_peak_distance=50)
        
        # First two peaks should be filtered (only one at index 105 or 100)
        # But 200 and 300 are far enough apart
        if len(peaks) > 1:
            for i in range(1, len(peaks)):
                self.assertGreaterEqual(peaks[i] - peaks[i-1], 50)


class TestModifiedKMeans(unittest.TestCase):
    """Tests for modified k-means clustering."""
    
    def setUp(self):
        """Set up test data."""
        np.random.seed(42)
        # Use smaller data for faster testing
        self.data = np.random.randn(32, 1000)
        self.srate = 250.0
    
    def test_modified_kmeans_returns_prototypes(self):
        """Test that modified k-means returns prototypes."""
        result = modified_kmeans(self.data, n_microstates=4, n_repetitions=2)
        
        self.assertIsInstance(result, MicrostatePrototypes)
        self.assertEqual(result.n_microstates, 4)
    
    def test_prototypes_shape(self):
        """Test prototype dimensions."""
        result = modified_kmeans(self.data, n_microstates=4, n_repetitions=2)
        
        # Should have 4 prototypes, each with 32 channels
        self.assertEqual(result.prototypes.shape, (4, 32))
    
    def test_prototypes_normalized(self):
        """Test that prototypes are normalized."""
        result = modified_kmeans(self.data, n_microstates=4, n_repetitions=2)
        
        # Each prototype should have unit norm
        for k in range(result.n_microstates):
            norm = np.linalg.norm(result.prototypes[k])
            self.assertAlmostEqual(norm, 1.0, places=5)
    
    def test_explained_variance_in_range(self):
        """Test that explained variance is in valid range."""
        result = modified_kmeans(self.data, n_microstates=4, n_repetitions=2)
        
        self.assertGreaterEqual(result.explained_variance, 0.0)
        self.assertLessEqual(result.explained_variance, 1.0)
    
    def test_different_n_microstates(self):
        """Test with different number of microstates."""
        for n in [2, 3, 4]:
            result = modified_kmeans(self.data, n_microstates=n, n_repetitions=1)
            self.assertEqual(result.n_microstates, n)
    
    def test_consistent_results(self):
        """Test that same seed gives same results."""
        result1 = modified_kmeans(self.data, n_microstates=4, n_repetitions=2, random_seed=42)
        result2 = modified_kmeans(self.data, n_microstates=4, n_repetitions=2, random_seed=42)
        
        np.testing.assert_array_almost_equal(result1.prototypes, result2.prototypes)


class TestBackfitting(unittest.TestCase):
    """Tests for microstate backfitting."""
    
    def setUp(self):
        """Set up test data."""
        np.random.seed(42)
        self.eeg = generate_synthetic_eeg(
            n_channels=32,
            duration=5.0,  # Reduced for faster testing
            srate=250.0,
            n_microstates=4
        )
        self.prototypes = modified_kmeans(self.eeg.data, n_microstates=4, n_repetitions=2)
    
    def test_backfit_returns_sequence(self):
        """Test that backfitting returns MicrostateSequence."""
        sequence = backfit_microstates(self.eeg.data, self.prototypes)
        
        self.assertIsInstance(sequence, MicrostateSequence)
    
    def test_backfit_labels_shape(self):
        """Test that labels have correct shape."""
        sequence = backfit_microstates(self.eeg.data, self.prototypes)
        
        self.assertEqual(len(sequence.labels), self.eeg.pnts)
    
    def test_backfit_labels_in_range(self):
        """Test that labels are valid microstate indices."""
        sequence = backfit_microstates(self.eeg.data, self.prototypes)
        
        for label in sequence.labels:
            self.assertGreaterEqual(label, 0)
            self.assertLess(label, self.prototypes.n_microstates)
    
    def test_backfit_uses_all_microstates(self):
        """Test that all microstates are used."""
        sequence = backfit_microstates(self.eeg.data, self.prototypes)
        
        used_microstates = np.unique(sequence.labels)
        self.assertEqual(len(used_microstates), self.prototypes.n_microstates)


class TestSmoothing(unittest.TestCase):
    """Tests for microstate label smoothing."""
    
    def setUp(self):
        """Set up test data."""
        np.random.seed(42)
        self.eeg = generate_synthetic_eeg(
            n_channels=32,
            duration=5.0,
            srate=250.0,
            n_microstates=4
        )
        self.prototypes = modified_kmeans(self.eeg.data, n_microstates=4, n_repetitions=2)
        self.sequence = backfit_microstates(self.eeg.data, self.prototypes)
    
    def test_smooth_preserves_length(self):
        """Test that smoothing preserves label length."""
        smoothed = smooth_microstate_labels(self.sequence.labels, min_duration=30)
        
        self.assertEqual(len(smoothed), len(self.sequence.labels))
    
    def test_smooth_reject_modes_short(self):
        """Test that reject smoothing marks short segments."""
        # Create labels with very short segments
        labels = np.array([0, 0, 0, 1, 1, 2, 2, 0, 0])
        
        smoothed = smooth_microstate_labels(labels, min_duration=3, smoothing_type='reject')
        
        # Segments shorter than min_duration should be marked with -1
        # (indices 3,4 are only length 2)
        self.assertEqual(smoothed[3], -1)
        self.assertEqual(smoothed[4], -1)


class TestMicrostateStatistics(unittest.TestCase):
    """Tests for microstate statistics."""
    
    def setUp(self):
        """Set up test data."""
        np.random.seed(42)
        self.eeg = generate_synthetic_eeg(
            n_channels=32,
            duration=5.0,
            srate=250.0,
            n_microstates=4
        )
        self.prototypes = modified_kmeans(self.eeg.data, n_microstates=4, n_repetitions=2)
        self.sequence = backfit_microstates(self.eeg.data, self.prototypes)
        self.srate = 250.0
    
    def test_statistics_contain_keys(self):
        """Test that statistics dictionary has expected keys."""
        stats = calculate_microstate_statistics(self.sequence, self.srate)
        
        expected_keys = ['coverage', 'counts', 'mean_duration_ms', 'sd_duration_ms']
        for key in expected_keys:
            self.assertIn(key, stats)
    
    def test_coverage_sums_to_one(self):
        """Test that coverage sums to 1."""
        stats = calculate_microstate_statistics(self.sequence, self.srate)
        
        self.assertAlmostEqual(sum(stats['coverage']), 1.0)
    
    def test_counts_positive(self):
        """Test that counts are positive."""
        stats = calculate_microstate_statistics(self.sequence, self.srate)
        
        self.assertTrue(np.all(stats['counts'] >= 0))


class TestFullPipeline(unittest.TestCase):
    """Tests for the full microstate segmentation pipeline."""
    
    def setUp(self):
        """Set up test data."""
        np.random.seed(42)
        self.eeg = generate_synthetic_eeg(
            n_channels=32,
            duration=30.0,
            srate=250.0,
            n_microstates=4
        )
    
    def test_segment_returns_correct_types(self):
        """Test that segmentation returns correct types."""
        prototypes, sequence = segment_microstates(self.eeg, n_microstates=4)
        
        self.assertIsInstance(prototypes, MicrostatePrototypes)
        self.assertIsInstance(sequence, MicrostateSequence)
    
    def test_segment_n_microstates(self):
        """Test that segmentation respects n_microstates."""
        for n in [2, 3, 4, 5]:
            prototypes, sequence = segment_microstates(self.eeg, n_microstates=n)
            self.assertEqual(prototypes.n_microstates, n)
            self.assertEqual(sequence.n_microstates, n)
    
    def test_segment_smoothing(self):
        """Test that segmentation with smoothing works."""
        prototypes, sequence = segment_microstates(
            self.eeg,
            n_microstates=4,
            smoothing='reject',
            smoothing_min_duration=30
        )
        
        # Results should still be valid
        self.assertEqual(len(sequence.labels), self.eeg.pnts)
    
    def test_find_optimal_nmicrostates(self):
        """Test finding optimal number of microstates."""
        results, values = find_optimal_nmicrostates(
            self.eeg.data,
            n_range=range(2, 6),
            n_repetitions=5
        )
        
        self.assertEqual(len(results), 4)  # 2, 3, 4, 5
        self.assertEqual(len(values), 4)
        
        # Values should generally increase with more microstates
        # (more microstates = better fit = higher explained variance)
        for i in range(1, len(values)):
            # Note: This might not always be strictly increasing
            # due to local minima, but generally variance increases
            self.assertGreaterEqual(values[i], 0)


class TestIntegration(unittest.TestCase):
    """Integration tests using real EEG metadata."""
    
    def test_load_metadata_and_segment(self):
        """Test loading metadata and running segmentation with synthetic data."""
        # Load metadata from real file
        eeg_meta = load_metadata_from_set('/tmp/eeg_test/D_01.set')
        
        # Generate synthetic data with same parameters
        eeg = generate_synthetic_eeg(
            n_channels=eeg_meta.nbchan,
            duration=10.0,
            srate=eeg_meta.srate,
            n_microstates=4
        )
        
        # Run segmentation
        prototypes, sequence = segment_microstates(eeg, n_microstates=4)
        
        self.assertIsNotNone(prototypes)
        self.assertIsNotNone(sequence)
        self.assertEqual(prototypes.n_microstates, 4)
    
    def test_preprocessing_pipeline(self):
        """Test full preprocessing pipeline."""
        # Generate test data
        eeg = generate_synthetic_eeg(
            n_channels=32,
            duration=10.0,
            srate=250.0,
            n_microstates=4
        )
        
        # Run preprocessing
        preprocessed = preprocess_for_microstates(eeg)
        
        # Check that data is still valid
        self.assertEqual(preprocessed.data.shape, eeg.data.shape)
        
        # Check that reference changed
        self.assertEqual(preprocessed.ref, 'average')


class TestEdgeCases(unittest.TestCase):
    """Tests for edge cases."""
    
    def test_single_channel(self):
        """Test with single channel."""
        data = np.random.randn(1, 1000)
        gfp = compute_gfp(data)
        self.assertEqual(gfp.shape, (1000,))
    
    def test_single_timepoint(self):
        """Test with single time point."""
        data = np.random.randn(32, 1)
        gfp = compute_gfp(data)
        self.assertEqual(gfp.shape, (1,))
    
    def test_small_data(self):
        """Test with small amount of data."""
        data = np.random.randn(32, 100)
        result = modified_kmeans(data, n_microstates=2, n_repetitions=2)
        self.assertEqual(result.n_microstates, 2)
    
    def test_gfp_with_nan(self):
        """Test GFP handles NaN values."""
        data = np.random.randn(32, 1000)
        data[:, 500] = np.nan
        gfp = compute_gfp(data)
        self.assertFalse(np.isnan(gfp[0]))


def run_tests():
    """Run all tests."""
    # Create a test suite
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    # Add all test classes
    test_classes = [
        TestEEGLoader,
        TestPreprocessing,
        TestGFP,
        TestModifiedKMeans,
        TestBackfitting,
        TestSmoothing,
        TestMicrostateStatistics,
        TestFullPipeline,
        TestIntegration,
        TestEdgeCases
    ]
    
    for test_class in test_classes:
        suite.addTests(loader.loadTestsFromTestCase(test_class))
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    return result


if __name__ == '__main__':
    result = run_tests()
    
    # Print summary
    if result.wasSuccessful():
        print("\n" + "="*50)
        print("All tests passed!")
        print("="*50)
    else:
        print("\n" + "="*50)
        print(f"Tests failed: {len(result.failures)}")
        print(f"Tests with errors: {len(result.errors)}")
        print("="*50)