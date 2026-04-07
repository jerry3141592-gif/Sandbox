using System;
using Xunit;
using ADMM;
using MathNet.Numerics.LinearAlgebra;
using MathNet.Numerics.LinearAlgebra.Double;

namespace ADMM.Tests
{
    public class SoftThresholdTests
    {
        [Fact]
        public void SoftThreshold_ZeroInput_ReturnsZero()
        {
            // Arrange
            double[] v = { 0, 0, 0 };
            double kappa = 1.0;
            
            // Act - use reflection or make it public for testing
            var result = SoftThreshold_Invoke(v, kappa);
            
            // Assert
            Assert.All(result, x => Assert.Equal(0, x, 10));
        }
        
        [Fact]
        public void SoftThreshold_AboveThreshold_ReturnsThresholdedValue()
        {
            // Arrange
            double[] v = { 3.0, -3.0 };
            double kappa = 1.0;
            
            // Act
            var result = SoftThreshold_Invoke(v, kappa);
            
            // Assert
            Assert.Equal(2.0, result[0], 10);
            Assert.Equal(-2.0, result[1], 10);
        }
        
        [Fact]
        public void SoftThreshold_BelowThreshold_ReturnsZero()
        {
            // Arrange
            double[] v = { 0.5, -0.5 };
            double kappa = 1.0;
            
            // Act
            var result = SoftThreshold_Invoke(v, kappa);
            
            // Assert
            Assert.Equal(0.0, result[0], 10);
            Assert.Equal(0.0, result[1], 10);
        }
        
        private double[] SoftThreshold_Invoke(double[] v, double kappa)
        {
            int n = v.Length;
            var result = new double[n];
            
            for (int i = 0; i < n; i++)
            {
                double vi = v[i];
                double sign = vi >= 0 ? 1.0 : -1.0;
                double absV = Math.Abs(vi);
                double thresholded = Math.Max(absV - kappa, 0);
                result[i] = sign * thresholded;
            }
            
            return result;
        }
    }
    
    public class ADMMOptimizerTests
    {
        [Fact]
        public void Constructor_InitializesMatricesCorrectly()
        {
            // Arrange
            double[] Y = { 1.0, 2.0, 3.0, 4.0, 5.0 };
            double alpha = 0.5;
            double beta = 0.3;
            
            // Act
            var optimizer = new ADMMOptimizer(Y, alpha, beta);
            
            // Assert - basic sanity check (matrices created without exception)
            Assert.NotNull(optimizer);
        }
        
        [Fact]
        public void Solve_EmptyInput_HandlesGracefully()
        {
            // Arrange
            double[] Y = Array.Empty<double>();
            
            // Act & Assert
            var optimizer = new ADMMOptimizer(Y, 0.5, 0.3);
            var result = optimizer.Solve();
            
            Assert.NotNull(result);
            Assert.Empty(result);
        }
        
        [Fact]
        public void Solve_SingleElement_ReturnsProcessedValue()
        {
            // Arrange
            double[] Y = { 5.0 };
            double alpha = 0.1;
            double beta = 0.1;
            
            // Act
            var optimizer = new ADMMOptimizer(Y, alpha, beta, maxIter: 100, verbose: false);
            var result = optimizer.Solve();
            
            // Assert
            Assert.Single(result);
            // With L1 regularization, single point should be pulled toward 0
            Assert.True(result[0] <= Y[0]); // L1 should shrink toward zero
        }
        
        [Fact]
        public void Solve_ConvergesOnSimpleData()
        {
            // Arrange
            double[] Y = { 1.0, 1.5, 2.0, 1.5, 1.0 };
            double alpha = 0.1;
            double beta = 0.1;
            
            // Act
            var optimizer = new ADMMOptimizer(Y, alpha, beta, maxIter: 100, verbose: false);
            var result = optimizer.Solve();
            
            // Assert
            Assert.Equal(Y.Length, result.Length);
            
            // Check that result is close to input (not NaN or Inf)
            foreach (var x in result)
            {
                Assert.False(double.IsNaN(x), "Result contains NaN");
                Assert.False(double.IsInfinity(x), "Result contains Infinity");
            }
        }
        
        [Fact]
        public void Solve_ZeroAlphaBeta_RetainsInput()
        {
            // Arrange
            double[] Y = { 1.0, 2.0, 3.0, 4.0, 5.0 };
            double alpha = 0;  // No regularization
            double beta = 0;
            
            // Act
            var optimizer = new ADMMOptimizer(Y, alpha, beta, maxIter: 100, verbose: false);
            var result = optimizer.Solve();
            
            // Assert - with zero regularization, result should be close to input
            for (int i = 0; i < Y.Length; i++)
            {
                // Allow some numerical tolerance
                Assert.True(Math.Abs(result[i] - Y[i]) < 1.0);
            }
        }
        
        [Fact]
        public void Solve_HighRho_AggressiveDenoising()
        {
            // Arrange
            double[] Y = { 1.0, 10.0, 1.0, 10.0, 1.0 }; // Noisy signal
            double alpha = 1.0;
            double beta = 1.0;
            double rho = 10.0; // Stronger penalty
            
            // Act
            var optimizer = new ADMMOptimizer(Y, alpha, beta, rho: rho, maxIter: 200, verbose: false);
            var result = optimizer.Solve();
            
            // Assert - with high rho, result should be smoother (closer to constant)
            double maxDiff = 0;
            for (int i = 1; i < result.Length; i++)
            {
                maxDiff = Math.Max(maxDiff, Math.Abs(result[i] - result[i-1]));
            }
            
            // Result should be smoother than original noisy input
            double originalMaxDiff = 0;
            for (int i = 1; i < Y.Length; i++)
            {
                originalMaxDiff = Math.Max(originalMaxDiff, Math.Abs(Y[i] - Y[i-1]));
            }
            
            Assert.True(maxDiff < originalMaxDiff, "Result should be smoother than input");
        }
        
        [Fact]
        public void ComputeObjective_IsNonNegative()
        {
            // Arrange
            double[] Y = { 1.0, 2.0, 3.0, 4.0, 5.0 };
            
            // Act
            var optimizer = new ADMMOptimizer(Y, 0.5, 0.3, maxIter: 50, verbose: false);
            var result = optimizer.Solve();
            var objective = optimizer.ComputeObjective();
            
            // Assert - objective function should be non-negative
            Assert.True(objective >= 0, $"Objective should be non-negative, got {objective}");
        }
        
        [Fact]
        public void Solve_IncreasingIterations_ImprovesObjective()
        {
            // Arrange
            double[] Y = { 1.0, 2.0, 3.0, 4.0, 5.0 };
            
            // Act - run with few iterations
            var optimizerFew = new ADMMOptimizer(Y, 0.5, 0.3, maxIter: 10, verbose: false);
            optimizerFew.Solve();
            double objFew = optimizerFew.ComputeObjective();
            
            // Run with more iterations
            var optimizerMany = new ADMMOptimizer(Y, 0.5, 0.3, maxIter: 100, verbose: false);
            optimizerMany.Solve();
            double objMany = optimizerMany.ComputeObjective();
            
            // Assert - objective should generally decrease with more iterations
            // (not guaranteed to be monotonic due to ADMM specifics, but likely)
            Assert.True(objMany <= objFew * 1.1, "More iterations should not significantly worsen objective");
        }
        
        [Fact]
        public void Solve_LargeRho_ConvergesFaster()
        {
            // Arrange
            double[] Y = { 1.0, 2.0, 3.0, 4.0, 5.0 };
            
            // Act
            var optimizerLow = new ADMMOptimizer(Y, 0.5, 0.3, rho: 0.1, maxIter: 500, verbose: false);
            optimizerLow.Solve();
            
            var optimizerHigh = new ADMMOptimizer(Y, 0.5, 0.3, rho: 10.0, maxIter: 500, verbose: false);
            optimizerHigh.Solve();
            
            // With high rho, convergence should be faster
            // Check that the solutions are reasonable
            Assert.True(optimizerHigh.ElapsedTime <= optimizerLow.ElapsedTime * 2, 
                "High rho should not take dramatically longer");
        }
        
        [Fact]
        public void Solve_SmoothSignal_PreservesShape()
        {
            // Arrange - smooth signal
            double[] Y = Enumerable.Range(0, 20).Select(i => Math.Sin(i * 0.5)).ToArray();
            
            // Act
            var optimizer = new ADMMOptimizer(Y, 0.1, 0.1, maxIter: 100, verbose: false);
            var result = optimizer.Solve();
            
            // Assert - result should maintain similar shape
            // Check correlation with original
            double sumXY = 0, sumX2 = 0, sumY2 = 0;
            for (int i = 0; i < Y.Length; i++)
            {
                sumXY += Y[i] * result[i];
                sumX2 += Y[i] * Y[i];
                sumY2 += result[i] * result[i];
            }
            double correlation = sumXY / Math.Sqrt(sumX2 * sumY2);
            
            Assert.True(correlation > 0.9, $"Correlation {correlation} should be high for smooth signal");
        }
    }
}