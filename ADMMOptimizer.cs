using System;
using System.Linq;
using MathNet.Numerics.LinearAlgebra;
using MathNet.Numerics.LinearAlgebra.Double;

namespace ADMM
{
    /// <summary>
    /// Alternating Direction Method of Multipliers (ADMM) for signal denoising
    /// Solves: min |X - Y|_1 + alpha * |D2 X|_1 + beta * |D3 X|_1
    /// </summary>
    public class ADMMOptimizer
    {
        private Vector<double> _Y = null!;
        private int _N;
        private double _alpha;
        private double _beta;
        private double _h;
        private double _rho;
        private int _maxIter;
        private double _tolAbs;
        private double _tolRel;
        private bool _verbose;
        
        private SparseMatrix _D2 = null!;
        private SparseMatrix _D3 = null!;
        
        private Vector<double> _X = null!;
        private Vector<double> _Z1 = null!;
        private Vector<double> _Z2 = null!;
        private Vector<double> _Z3 = null!;
        private Vector<double> _U1 = null!;
        private Vector<double> _U2 = null!;
        private Vector<double> _U3 = null!;
        
        private SparseMatrix _A = null!;
        private MathNet.Numerics.LinearAlgebra.Factorization.LU<double> _factorization = null!;
        private double _elapsedTime;
        
        public double ElapsedTime => _elapsedTime;
        
        public ADMMOptimizer(double[] Y, double alpha, double beta, 
                             double h = 1.0, double rho = 1.0, 
                             int maxIter = 500, double tolAbs = 1e-6, 
                             double tolRel = 1e-6, bool verbose = true)
        {
            _Y = Vector<double>.Build.DenseOfArray(Y.Select(x => (double)x).ToArray());
            _N = _Y.Count;
            _alpha = alpha;
            _beta = beta;
            _h = h;
            _rho = rho;
            _maxIter = maxIter;
            _tolAbs = tolAbs;
            _tolRel = tolRel;
            _verbose = verbose;
            
            if (_N > 0)
            {
                _buildMatrices();
                _initVariables();
                _precomputeA();
            }
        }
        
        private void _buildMatrices()
        {
            int N = _N;
            double h2 = _h * _h;
            double h3 = h2 * _h;
            
            // D2: second order finite difference matrix
            _D2 = SparseMatrix.Create(N, N, 0.0);
            
            if (N >= 2)
            {
                // Interior points: central difference
                for (int i = 1; i < N - 1; i++)
                {
                    _D2[i, i - 1] = 1.0 / h2;
                    _D2[i, i] = -2.0 / h2;
                    _D2[i, i + 1] = 1.0 / h2;
                }
                
                // Forward difference at start: X''_0 ≈ 0
                _D2[0, 0] = -2.0 / h2;
                _D2[0, 1] = 2.0 / h2;
                
                // Backward difference at end: X''_{N-1} ≈ 0
                _D2[N - 1, N - 2] = 2.0 / h2;
                _D2[N - 1, N - 1] = -2.0 / h2;
            }
            else if (N == 1)
            {
                // Single element: use simple approximation
                _D2[0, 0] = 1.0;
            }
            
            // D3: third order finite difference matrix
            _D3 = SparseMatrix.Create(N, N, 0.0);
            
            if (N >= 4)
            {
                // Interior points
                for (int i = 2; i < N - 2; i++)
                {
                    _D3[i, i - 2] = -1.0 / h3;
                    _D3[i, i - 1] = 3.0 / h3;
                    _D3[i, i] = -3.0 / h3;
                    _D3[i, i + 1] = 1.0 / h3;
                }
                
                // Forward difference at start: X'''_0 ≈ 0
                _D3[0, 0] = 2.0 / h3;
                _D3[0, 1] = -5.0 / h3;
                _D3[0, 2] = 4.0 / h3;
                _D3[0, 3] = -1.0 / h3;
                
                // Backward difference at end: X'''_{N-1} ≈ 0
                _D3[N - 1, N - 4] = -1.0 / h3;
                _D3[N - 1, N - 3] = 4.0 / h3;
                _D3[N - 1, N - 2] = -5.0 / h3;
                _D3[N - 1, N - 1] = 2.0 / h3;
            }
            else if (N >= 2)
            {
                // For smaller N, use simpler difference approximations
                _D3[0, 0] = 1.0;
                if (N > 1)
                    _D3[0, 1] = -1.0;
            }
        }
        
        private void _initVariables()
        {
            _X = _Y.Clone();
            _Z1 = Vector<double>.Build.Dense(_N);
            _Z2 = Vector<double>.Build.Dense(_D2.RowCount);
            _Z3 = Vector<double>.Build.Dense(_D3.RowCount);
            _U1 = Vector<double>.Build.Dense(_N);
            _U2 = Vector<double>.Build.Dense(_D2.RowCount);
            _U3 = Vector<double>.Build.Dense(_D3.RowCount);
        }
        
        private void _precomputeA()
        {
            // A = I + rho * (D2^T * D2 + D3^T * D3)
            var I = Matrix<double>.Build.DenseIdentity(_N);
            var D2tD2 = _D2.Transpose() * _D2;
            var D3tD3 = _D3.Transpose() * _D3;
            
            // Convert to sparse using SparseMatrix.Create
            var denseA = I + _rho * (D2tD2 + D3tD3);
            _A = SparseMatrix.Create(_N, _N, 0.0);
            
            // Copy dense to sparse
            for (int i = 0; i < _N; i++)
            {
                for (int j = 0; j < _N; j++)
                {
                    _A[i, j] = denseA[i, j];
                }
            }
            
            // LU decomposition for efficient solving
            _factorization = _A.LU();
        }
        
        /// <summary>
        /// Soft thresholding operator (shrinkage)
        /// </summary>
        private static Vector<double> SoftThreshold(Vector<double> v, double kappa)
        {
            int n = v.Count;
            var result = Vector<double>.Build.Dense(n);
            
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
        
        /// <summary>
        /// Compute the objective function value
        /// </summary>
        public double ComputeObjective()
        {
            double obj = 0.0;
            
            // |X - Y|_1
            for (int i = 0; i < _N; i++)
            {
                obj += Math.Abs(_X[i] - _Y[i]);
            }
            
            // alpha * |D2 X|_1
            if (_alpha > 0)
            {
                var D2X = _D2 * _X;
                for (int i = 0; i < D2X.Count; i++)
                {
                    obj += _alpha * Math.Abs(D2X[i]);
                }
            }
            
            // beta * |D3 X|_1
            if (_beta > 0)
            {
                var D3X = _D3 * _X;
                for (int i = 0; i < D3X.Count; i++)
                {
                    obj += _beta * Math.Abs(D3X[i]);
                }
            }
            
            return obj;
        }
        
        /// <summary>
        /// Run the ADMM optimization
        /// </summary>
        public double[] Solve()
        {
            if (_N == 0)
            {
                return Array.Empty<double>();
            }
            
            var startTime = DateTime.UtcNow;
            
            for (int k = 0; k < _maxIter; k++)
            {
                // Store old values for convergence check
                Vector<double>? Z1_old = null;
                Vector<double>? Z2_old = null;
                Vector<double>? Z3_old = null;
                if (k > 0)
                {
                    Z1_old = _Z1.Clone();
                    Z2_old = _Z2.Clone();
                    Z3_old = _Z3.Clone();
                }
                
                // ========== X-step (solve linear system) ==========
                var rhs = (_Y + _Z1 - _U1) + 
                          _D2.Transpose() * (_Z2 - _U2) + 
                          _D3.Transpose() * (_Z3 - _U3);
                
                // Solve: A * X = RHS using LU factorization
                _X = _factorization.Solve(rhs);
                
                // ========== Z-step (soft thresholding) ==========
                var X_minus_Y_plus_U1 = _X - _Y + _U1;
                _Z1 = SoftThreshold(X_minus_Y_plus_U1, 1.0 / _rho);
                
                var D2X_plus_U2 = _D2 * _X + _U2;
                _Z2 = SoftThreshold(D2X_plus_U2, _alpha / _rho);
                
                var D3X_plus_U3 = _D3 * _X + _U3;
                _Z3 = SoftThreshold(D3X_plus_U3, _beta / _rho);
                
                // ========== U-step (dual variable update) ==========
                _U1 += (_X - _Y) - _Z1;
                _U2 += (_D2 * _X) - _Z2;
                _U3 += (_D3 * _X) - _Z3;
                
                // ========== Convergence check ==========
                if (k > 0 && k % 50 == 0)
                {
                    // Compute objective
                    double obj = ComputeObjective();
                    
                    // Primal residual
                    double r_norm = 0.0;
                    for (int i = 0; i < _N; i++)
                    {
                        double diff = _X[i] - _Y[i] - _Z1[i];
                        r_norm += diff * diff;
                    }
                    r_norm = Math.Sqrt(r_norm);
                    
                    // Dual residual
                    double s_norm = 0.0;
                    for (int i = 0; i < _N; i++)
                    {
                        double diff = _Z1[i] - Z1_old![i];
                        s_norm += diff * diff;
                    }
                    s_norm = _rho * Math.Sqrt(s_norm);
                    
                    if (_verbose)
                    {
                        Console.WriteLine($"Iter {k,4}: obj={obj:F4}, r={r_norm:E2}, s={s_norm:E2}");
                    }
                    
                    // Stopping criterion
                    double eps = _tolAbs + _tolRel * Math.Max(_X.L2Norm(), _Z1.L2Norm());
                    if (r_norm < eps && s_norm < eps)
                    {
                        if (_verbose)
                        {
                            Console.WriteLine($"\n✓ Converged at iteration {k}");
                        }
                        break;
                    }
                }
            }
            
            _elapsedTime = (DateTime.UtcNow - startTime).TotalSeconds;
            
            if (_verbose)
            {
                Console.WriteLine($"Time: {_elapsedTime:F2}s");
            }
            
            return _X.ToArray();
        }
    }
}