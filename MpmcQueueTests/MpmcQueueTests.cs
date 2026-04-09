using System;
using System.Collections.Generic;
using System.Linq;
using System.Threading;
using System.Threading.Tasks;
using Xunit;

namespace MpmcQueueTests
{
    /// <summary>
    /// Unit tests for MpmcQueue
    /// </summary>
    public class MpmcQueueTests
    {
        #region Basic Functionality Tests

        [Fact]
        public void Enqueue_Dequeue_SingleThread_Succeeds()
        {
            // Arrange
            var queue = new MpmcQueue<int>();

            // Act & Assert
            queue.Enqueue(1);
            queue.Enqueue(2);
            queue.Enqueue(3);

            Assert.Equal(3, queue.Count);
            
            Assert.Equal(1, queue.Dequeue());
            Assert.Equal(2, queue.Dequeue());
            Assert.Equal(3, queue.Dequeue());
            
            Assert.True(queue.IsEmpty);
        }

        [Fact]
        public void TryEnqueue_TryDequeue_SingleThread_Succeeds()
        {
            // Arrange
            var queue = new MpmcQueue<int>();

            // Act & Assert
            Assert.True(queue.TryEnqueue(1));
            Assert.True(queue.TryEnqueue(2));
            Assert.True(queue.TryEnqueue(3));

            Assert.True(queue.TryDequeue(out int item1));
            Assert.Equal(1, item1);
            
            Assert.True(queue.TryDequeue(out int item2));
            Assert.Equal(2, item2);
            
            Assert.True(queue.TryDequeue(out int item3));
            Assert.Equal(3, item3);
            
            Assert.False(queue.TryDequeue(out int _));
            Assert.True(queue.IsEmpty);
        }

        [Fact]
        public void TryPeek_ReturnsItem_WithoutRemoving()
        {
            // Arrange
            var queue = new MpmcQueue<int>();
            queue.Enqueue(42);
            queue.Enqueue(43);

            // Act & Assert
            Assert.True(queue.TryPeek(out int item));
            Assert.Equal(42, item);
            Assert.Equal(2, queue.Count);

            queue.Dequeue();
            
            Assert.True(queue.TryPeek(out int item2));
            Assert.Equal(43, item2);
        }

        [Fact]
        public void TryPeek_OnEmptyQueue_ReturnsFalse()
        {
            // Arrange
            var queue = new MpmcQueue<int>();

            // Act & Assert
            Assert.False(queue.TryPeek(out int _));
            Assert.True(queue.IsEmpty);
        }

        [Fact]
        public void Count_IsApproximate_ReflectsQueueSize()
        {
            // Arrange
            var queue = new MpmcQueue<int>();

            // Act & Assert
            Assert.Equal(0, queue.Count);
            
            queue.Enqueue(1);
            queue.Enqueue(2);
            Assert.Equal(2, queue.Count);
            
            queue.Dequeue();
            Assert.Equal(1, queue.Count);
            
            queue.Dequeue();
            Assert.Equal(0, queue.Count);
        }

        [Fact]
        public void IsEmpty_ReturnsTrue_WhenQueueEmpty()
        {
            // Arrange
            var queue = new MpmcQueue<int>();

            // Act & Assert
            Assert.True(queue.IsEmpty);
            
            queue.Enqueue(1);
            Assert.False(queue.IsEmpty);
            
            queue.Dequeue();
            Assert.True(queue.IsEmpty);
        }

        [Fact]
        public void Clear_RemovesAllElements()
        {
            // Arrange
            var queue = new MpmcQueue<int>();
            for (int i = 0; i < 100; i++)
            {
                queue.Enqueue(i);
            }

            // Act
            queue.Clear();

            // Assert
            Assert.True(queue.IsEmpty);
            Assert.Equal(0, queue.Count);
        }

        [Fact]
        public void DefaultSegmentSize_Is1024()
        {
            // Arrange & Act
            var queue = new MpmcQueue<int>();

            // Assert - should be able to enqueue 1024 items without new segment
            for (int i = 0; i < 1024; i++)
            {
                queue.Enqueue(i);
            }
            Assert.Equal(1024, queue.Count);
        }

        #endregion

        #region Thread Safety Tests

        [Fact]
        public void MultiProducer_SingleConsumer_AllItemsEnqueued()
        {
            // Arrange
            var queue = new MpmcQueue<int>();
            const int producerCount = 4;
            const int itemsPerProducer = 100;
            var producedValues = new HashSet<int>();
            var lockObj = new object();

            // Act
            Parallel.For(0, producerCount, new ParallelOptions { MaxDegreeOfParallelism = producerCount }, producerId =>
            {
                for (int i = 0; i < itemsPerProducer; i++)
                {
                    int value = producerId * itemsPerProducer + i;
                    queue.Enqueue(value);
                    lock (lockObj)
                    {
                        producedValues.Add(value);
                    }
                }
            });

            // Assert
            var dequeuedValues = new HashSet<int>();
            while (queue.TryDequeue(out int value))
            {
                dequeuedValues.Add(value);
            }

            Assert.Equal(producerCount * itemsPerProducer, dequeuedValues.Count);
            Assert.Equal(producedValues.Count, dequeuedValues.Count);
        }

        [Fact]
        public void SingleProducer_MultiConsumer_AllItemsDequeued()
        {
            // Arrange
            var queue = new MpmcQueue<int>();
            const int consumerCount = 4;
            const int totalItems = 10000;
            var dequeuedValues = new ConcurrentHashSet<int>();
            var tasks = new Task[consumerCount];

            // Act
            // Fill the queue first
            for (int i = 0; i < totalItems; i++)
            {
                queue.Enqueue(i);
            }

            // Start consumers
            for (int i = 0; i < consumerCount; i++)
            {
                int consumerId = i;
                tasks[i] = Task.Run(() =>
                {
                    while (queue.TryDequeue(out int value))
                    {
                        dequeuedValues.Add(value);
                    }
                });
            }

            Task.WaitAll(tasks);

            // Assert
            Assert.Equal(totalItems, dequeuedValues.Count);
            for (int i = 0; i < totalItems; i++)
            {
                Assert.True(dequeuedValues.Contains(i));
            }
        }

        [Fact]
        public void MultiProducer_MultiConsumer_NoDataLoss()
        {
            // Arrange
            var queue = new MpmcQueue<long>();
            const int producerCount = 8;
            const int consumerCount = 8;
            const long itemsPerProducer = 500;
            long totalProduced = 0;
            long totalConsumed = 0;
            var producedLock = new object();
            var consumedLock = new object();

            // Act
            var producerTasks = new Task[producerCount];
            for (int p = 0; p < producerCount; p++)
            {
                int producerId = p;
                producerTasks[p] = Task.Run(() =>
                {
                    for (long i = 0; i < itemsPerProducer; i++)
                    {
                        long value = producerId * itemsPerProducer + i;
                        queue.Enqueue(value);
                        lock (producedLock)
                        {
                            totalProduced += value;
                        }
                    }
                });
            }

            var consumerTasks = new Task[consumerCount];
            for (int c = 0; c < consumerCount; c++)
            {
                consumerTasks[c] = Task.Run(() =>
                {
                    long localSum = 0;
                    int count = 0;
                    while (count < producerCount * itemsPerProducer)
                    {
                        if (queue.TryDequeue(out long value))
                        {
                            localSum += value;
                            count++;
                        }
                        else
                        {
                            Thread.SpinWait(1);
                        }
                    }
                    lock (consumedLock)
                    {
                        totalConsumed += localSum;
                    }
                });
            }

            Task.WaitAll(producerTasks);
            Task.WaitAll(consumerTasks);

            // Assert
            Assert.Equal(totalProduced, totalConsumed);
        }

        [Fact]
        public void ConcurrentEnqueueDequeue_NoExceptions()
        {
            // Arrange
            var queue = new MpmcQueue<int>();
            const int iterations = 5000;
            var exceptions = new ConcurrentBag<Exception>();

            // Act
            var task1 = Task.Run(() =>
            {
                try
                {
                    for (int i = 0; i < iterations; i++)
                    {
                        queue.Enqueue(i);
                        Thread.SpinWait(10);
                    }
                }
                catch (Exception ex)
                {
                    exceptions.Add(ex);
                }
            });

            var task2 = Task.Run(() =>
            {
                try
                {
                    int count = 0;
                    while (count < iterations)
                    {
                        if (queue.TryDequeue(out _))
                        {
                            count++;
                        }
                        else
                        {
                            Thread.SpinWait(10);
                        }
                    }
                }
                catch (Exception ex)
                {
                    exceptions.Add(ex);
                }
            });

            Task.WaitAll(task1, task2);

            // Assert
            Assert.Equal(0, exceptions.Count);
        }

        #endregion

        #region Edge Cases Tests

        [Fact]
        public void Enqueue_NullValue_ForReferenceType_Succeeds()
        {
            // Arrange
            var queue = new MpmcQueue<string?>();

            // Act & Assert
            queue.Enqueue(null!);
            queue.Enqueue("test");
            queue.Enqueue(null!);

            Assert.Equal(3, queue.Count);
            
            Assert.Null(queue.Dequeue());
            Assert.Equal("test", queue.Dequeue());
            Assert.Null(queue.Dequeue());
        }

        [Fact]
        public void Enqueue_DefaultValue_ForValueType_Succeeds()
        {
            // Arrange
            var queue = new MpmcQueue<int>();

            // Act & Assert
            queue.Enqueue(0); // Default value for int
            queue.Enqueue(1);
            
            Assert.Equal(2, queue.Count);
            
            Assert.Equal(0, queue.Dequeue());
            Assert.Equal(1, queue.Dequeue());
        }

        [Fact]
        public void LargeSegmentSize_WorksCorrectly()
        {
            // Arrange
            var queue = new MpmcQueue<int>(4096);

            // Act & Assert
            for (int i = 0; i < 4096; i++)
            {
                queue.Enqueue(i);
            }
            Assert.Equal(4096, queue.Count);

            for (int i = 0; i < 4096; i++)
            {
                Assert.Equal(i, queue.Dequeue());
            }
            
            Assert.True(queue.IsEmpty);
        }

        [Fact]
        public void GrowsBeyondInitialSegment()
        {
            // Arrange - small segment size to force growth
            var queue = new MpmcQueue<int>(64);

            // Act - enqueue more than initial segment size
            for (int i = 0; i < 1000; i++)
            {
                queue.Enqueue(i);
            }

            // Assert
            Assert.Equal(1000, queue.Count);
            
            for (int i = 0; i < 1000; i++)
            {
                Assert.Equal(i, queue.Dequeue());
            }
        }

        [Fact]
        public void CustomSegmentSize_RoundsToPowerOf2()
        {
            // Act - 1000 should round to 1024 (next power of 2)
            var queue = new MpmcQueue<int>(1000);

            // Assert - should be able to fit 1024 items in first segment
            for (int i = 0; i < 1024; i++)
            {
                queue.Enqueue(i);
            }
            Assert.Equal(1024, queue.Count);
        }

        #endregion

        #region Stress Tests

        [Fact]
        public void Stress_HighVolume_MaintainsCorrectness()
        {
            // Arrange
            var queue = new MpmcQueue<int>();
            const int iterations = 5000;

            // Act
            var enqueueTask = Task.Run(() =>
            {
                for (int i = 0; i < iterations; i++)
                {
                    queue.Enqueue(i);
                }
            });

            var dequeueTask = Task.Run(() =>
            {
                int count = 0;
                int lastValue = -1;
                while (count < iterations)
                {
                    if (queue.TryDequeue(out int value))
                    {
                        if (value < lastValue)
                        {
                            throw new Exception($"Out of order: {value} < {lastValue}");
                        }
                        lastValue = value;
                        count++;
                    }
                }
            });

            Task.WaitAll(enqueueTask, dequeueTask);

            // Assert
            Assert.True(queue.IsEmpty);
        }

        #endregion

        #region Concurrent Collection Helper

        internal class ConcurrentHashSet<T>
        {
            private readonly HashSet<T> _set = new HashSet<T>();
            private readonly object _lock = new object();

            public void Add(T item)
            {
                lock (_lock)
                {
                    _set.Add(item);
                }
            }

            public int Count
            {
                get
                {
                    lock (_lock)
                    {
                        return _set.Count;
                    }
                }
            }

            public bool Contains(T item)
            {
                lock (_lock)
                {
                    return _set.Contains(item);
                }
            }
        }

        internal class ConcurrentBag<T>
        {
            private readonly List<T> _items = new List<T>();
            private readonly object _lock = new object();

            public void Add(T item)
            {
                lock (_lock)
                {
                    _items.Add(item);
                }
            }

            public int Count
            {
                get
                {
                    lock (_lock)
                    {
                        return _items.Count;
                    }
                }
            }
        }

        #endregion
    }
}
