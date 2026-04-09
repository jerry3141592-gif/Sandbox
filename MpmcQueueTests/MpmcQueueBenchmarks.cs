using System;
using System.Collections.Concurrent;
using System.Collections.Generic;
using System.Linq;
using System.Threading;
using System.Threading.Tasks;
using BenchmarkDotNet.Attributes;
using BenchmarkDotNet.Configs;
using BenchmarkDotNet.Jobs;
using BenchmarkDotNet.Running;

namespace MpmcQueueTests
{
    /// <summary>
    /// Benchmarks for MpmcQueue comparing with .NET ConcurrentQueue
    /// </summary>
    [Config(typeof(Config))]
    public class MpmcQueueBenchmarks
    {
        private class Config : ManualConfig
        {
            public Config()
            {
                AddJob(Job.Default
                    .WithWarmupCount(2)
                    .WithIterationCount(5)
                    .WithInvocationCount(1000000)
                    .WithUnrollFactor(1));
            }
        }

        private MpmcQueue<int> _mpmcQueue = null!;
        private ConcurrentQueue<int> _concurrentQueue = null!;
        
        private const int SingleProducerIterations = 1000000;
        private const int MultiProducerIterations = 100000;
        private const int ProducerCount = 4;
        private const int ConsumerCount = 4;

        [GlobalSetup]
        public void Setup()
        {
            _mpmcQueue = new MpmcQueue<int>(1024);
            _concurrentQueue = new ConcurrentQueue<int>();
        }

        #region Single Producer Benchmarks

        [Benchmark(Baseline = true)]
        public void ConcurrentQueue_SingleProducer_Enqueue()
        {
            var queue = _concurrentQueue;
            for (int i = 0; i < SingleProducerIterations; i++)
            {
                queue.Enqueue(i);
            }
        }

        [Benchmark]
        public void MpmcQueue_SingleProducer_Enqueue()
        {
            var queue = _mpmcQueue;
            for (int i = 0; i < SingleProducerIterations; i++)
            {
                queue.Enqueue(i);
            }
        }

        [Benchmark(Baseline = true)]
        public int ConcurrentQueue_SingleProducer_EnqueueDequeue()
        {
            var queue = _concurrentQueue;
            int result = 0;
            for (int i = 0; i < SingleProducerIterations; i++)
            {
                queue.Enqueue(i);
                if (queue.TryDequeue(out int item))
                {
                    result += item;
                }
            }
            return result;
        }

        [Benchmark]
        public int MpmcQueue_SingleProducer_EnqueueDequeue()
        {
            var queue = _mpmcQueue;
            int result = 0;
            for (int i = 0; i < SingleProducerIterations; i++)
            {
                queue.Enqueue(i);
                if (queue.TryDequeue(out int item))
                {
                    result += item;
                }
            }
            return result;
        }

        #endregion

        #region Multi-Threaded Benchmarks

        [Benchmark(Baseline = true)]
        public long ConcurrentQueue_MultiProducer_MultiConsumer()
        {
            var queue = new ConcurrentQueue<int>();
            var produced = new long[ProducerCount];
            var consumed = new long[ConsumerCount];
            
            var producerTasks = new Task[ProducerCount];
            for (int p = 0; p < ProducerCount; p++)
            {
                int producerId = p;
                producerTasks[p] = Task.Run(() =>
                {
                    for (int i = 0; i < MultiProducerIterations; i++)
                    {
                        queue.Enqueue(producerId * MultiProducerIterations + i);
                        produced[producerId]++;
                    }
                });
            }

            var consumerTasks = new Task[ConsumerCount];
            for (int c = 0; c < ConsumerCount; c++)
            {
                consumerTasks[c] = Task.Run(() =>
                {
                    int localConsumed = 0;
                    while (localConsumed < ProducerCount * MultiProducerIterations)
                    {
                        if (queue.TryDequeue(out int _))
                        {
                            localConsumed++;
                        }
                    }
                });
            }

            Task.WaitAll(producerTasks);
            Task.WaitAll(consumerTasks);
            
            return ProducerCount * MultiProducerIterations;
        }

        [Benchmark]
        public long MpmcQueue_MultiProducer_MultiConsumer()
        {
            var queue = new MpmcQueue<int>(1024);
            
            var producerTasks = new Task[ProducerCount];
            for (int p = 0; p < ProducerCount; p++)
            {
                int producerId = p;
                producerTasks[p] = Task.Run(() =>
                {
                    for (int i = 0; i < MultiProducerIterations; i++)
                    {
                        queue.Enqueue(producerId * MultiProducerIterations + i);
                    }
                });
            }

            var consumerTasks = new Task[ConsumerCount];
            for (int c = 0; c < ConsumerCount; c++)
            {
                consumerTasks[c] = Task.Run(() =>
                {
                    int localConsumed = 0;
                    while (localConsumed < ProducerCount * MultiProducerIterations)
                    {
                        if (queue.TryDequeue(out int _))
                        {
                            localConsumed++;
                        }
                    }
                });
            }

            Task.WaitAll(producerTasks);
            Task.WaitAll(consumerTasks);
            
            return ProducerCount * MultiProducerIterations;
        }

        #endregion

        #region Throughput Benchmarks

        [Benchmark(Baseline = true)]
        public void ConcurrentQueue_PushPop_Throughput()
        {
            var queue = _concurrentQueue;
            var mre = new ManualResetEvent(false);
            int totalOps = 0;
            
            var producer = new Thread(() =>
            {
                for (int i = 0; i < 100000; i++)
                {
                    queue.Enqueue(i);
                }
                mre.Set();
            });

            var consumer = new Thread(() =>
            {
                int count = 0;
                while (!mre.WaitOne(0) || queue.Count > 0)
                {
                    if (queue.TryDequeue(out _))
                    {
                        count++;
                    }
                }
                Interlocked.Exchange(ref totalOps, count);
            });

            producer.Start();
            consumer.Start();
            producer.Join();
            consumer.Join();
        }

        [Benchmark]
        public void MpmcQueue_PushPop_Throughput()
        {
            var queue = _mpmcQueue;
            var mre = new ManualResetEvent(false);
            int totalOps = 0;
            
            var producer = new Thread(() =>
            {
                for (int i = 0; i < 100000; i++)
                {
                    queue.Enqueue(i);
                }
                mre.Set();
            });

            var consumer = new Thread(() =>
            {
                int count = 0;
                while (!mre.WaitOne(0) || queue.Count > 0)
                {
                    if (queue.TryDequeue(out _))
                    {
                        count++;
                    }
                }
                Interlocked.Exchange(ref totalOps, count);
            });

            producer.Start();
            consumer.Start();
            producer.Join();
            consumer.Join();
        }

        #endregion

        #region Latency Benchmarks

        [Benchmark(Baseline = true)]
        public bool ConcurrentQueue_TryDequeue_Latency()
        {
            var queue = _concurrentQueue;
            queue.Enqueue(1);
            return queue.TryDequeue(out int _);
        }

        [Benchmark]
        public bool MpmcQueue_TryDequeue_Latency()
        {
            var queue = _mpmcQueue;
            queue.Enqueue(1);
            return queue.TryDequeue(out int _);
        }

        [Benchmark(Baseline = true)]
        public int ConcurrentQueue_Enqueue_Latency()
        {
            var queue = _concurrentQueue;
            queue.Enqueue(1);
            return 1;
        }

        [Benchmark]
        public int MpmcQueue_Enqueue_Latency()
        {
            var queue = _mpmcQueue;
            queue.Enqueue(1);
            return 1;
        }

        #endregion
    }
}

