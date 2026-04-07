using System;
using ADMM;

class Program
{
    static void Main(string[] args)
    {
        Console.WriteLine("=== ADMM Optimizer Demo ===\n");

        // Example 1: Basic usage with simple signal
        Console.WriteLine("Example 1: Simple signal denoising");
        double[] Y1 = { 1.0, 2.0, 3.0, 4.0, 5.0, 4.0, 3.0, 2.0, 1.0 };
        double alpha1 = 0.5;
        double beta1 = 0.3;

        var optimizer1 = new ADMMOptimizer(Y1, alpha1, beta1, maxIter: 100, verbose: true);
        double[] result1 = optimizer1.Solve();

        Console.WriteLine("\nInput:  " + string.Join(", ", Y1));
        Console.WriteLine("Output: " + string.Join(", ", Array.ConvertAll(result1, x => x.ToString("F4"))));
        Console.WriteLine("Objective: " + optimizer1.ComputeObjective().ToString("F6"));
        Console.WriteLine();

        // Example 2: Noisy signal with strong regularization
        Console.WriteLine("Example 2: Noisy signal with strong regularization");
        double[] Y2 = { 1.0, 10.0, 1.0, 10.0, 1.0, 10.0, 1.0, 10.0 };
        double alpha2 = 1.0;
        double beta2 = 1.0;
        double rho2 = 5.0;

        var optimizer2 = new ADMMOptimizer(Y2, alpha2, beta2, rho: rho2, maxIter: 200, verbose: false);
        double[] result2 = optimizer2.Solve();

        Console.WriteLine("Input:  " + string.Join(", ", Y2));
        Console.WriteLine("Output: " + string.Join(", ", Array.ConvertAll(result2, x => x.ToString("F4"))));
        Console.WriteLine("Objective: " + optimizer2.ComputeObjective().ToString("F6"));
        Console.WriteLine();

        // Example 3: Smooth signal preservation
        Console.WriteLine("Example 3: Smooth signal (sine wave)");
        int n = 50;
        double[] Y3 = new double[n];
        for (int i = 0; i < n; i++)
        {
            Y3[i] = Math.Sin(i * 0.3) + (new Random(i).NextDouble() - 0.5) * 0.3; // Sine + noise
        }

        var optimizer3 = new ADMMOptimizer(Y3, alpha: 0.1, beta: 0.1, maxIter: 100, verbose: false);
        double[] result3 = optimizer3.Solve();

        Console.WriteLine("Input (first 10):  " + string.Join(", ", Array.ConvertAll(Y3.Take(10).ToArray(), x => x.ToString("F2"))));
        Console.WriteLine("Output (first 10): " + string.Join(", ", Array.ConvertAll(result3.Take(10).ToArray(), x => x.ToString("F2"))));
        Console.WriteLine("Objective: " + optimizer3.ComputeObjective().ToString("F6"));
        Console.WriteLine();

        // Example 4: Edge case - small signal
        Console.WriteLine("Example 4: Small signal (3 elements)");
        double[] Y4 = { 1.0, 2.0, 1.0 };
        
        var optimizer4 = new ADMMOptimizer(Y4, 0.1, 0.1, maxIter: 50, verbose: false);
        double[] result4 = optimizer4.Solve();

        Console.WriteLine("Input:  " + string.Join(", ", Y4));
        Console.WriteLine("Output: " + string.Join(", ", Array.ConvertAll(result4, x => x.ToString("F4"))));
        Console.WriteLine("Objective: " + optimizer4.ComputeObjective().ToString("F6"));

        Console.WriteLine("\n=== Demo Complete ===");
    }
}