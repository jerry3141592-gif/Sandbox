using System;
using System.Runtime.CompilerServices;
using System.Threading;

/// <summary>
/// High-performance lock-free unbounded MPMC (Multi-Producer Multi-Consumer) queue.
/// Uses segmented ring buffers with linked segments - truly unbounded, grows dynamically.
/// Based on the classic MPMC queue design with sequence numbers.
/// </summary>
/// <typeparam name="T">Element type</typeparam>
public sealed class MpmcQueue<T>
{
    /// <summary>
    /// Default capacity per segment.
    /// </summary>
    public const int DefaultSegmentSize = 1024;

    private readonly int _segmentSize;
    private readonly int _segmentSizeMask;
    
    // Head segment (for dequeue) and tail segment (for enqueue)
    private Segment _head;
    private Segment _tail;

    /// <summary>
    /// Creates an unbounded MPMC queue.
    /// </summary>
    /// <param name="segmentSize">Size of each segment (power of 2 recommended)</param>
    public MpmcQueue(int segmentSize = DefaultSegmentSize)
    {
        if (segmentSize < 1)
            throw new ArgumentOutOfRangeException(nameof(segmentSize), "Segment size must be positive");
        
        // Round up to power of 2
        int actualSize = 1;
        while (actualSize < segmentSize)
            actualSize <<= 1;
        
        _segmentSize = actualSize;
        _segmentSizeMask = actualSize - 1;
        
        // Create initial segment
        var initialSegment = new Segment(actualSize, 0);
        _head = initialSegment;
        _tail = initialSegment;
    }

    /// <summary>
    /// Enqueues an item. Returns true if successful.
    /// This method never fails - it will grow the queue indefinitely.
    /// </summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public bool TryEnqueue(T item)
    {
        var tail = _tail;
        long pos = Interlocked.Add(ref tail.EnqueuePos, 1) - 1;
        
        // Check if we need a new segment
        if (pos >= tail.BasePosition + _segmentSize)
        {
            Segment next = tail.Next;
            if (next == null)
            {
                // Try to create next segment
                next = new Segment(_segmentSize, tail.BasePosition + _segmentSize);
                Segment prev = Interlocked.CompareExchange(ref tail.Next, next, null);
                if (prev != null)
                {
                    // Another thread created it, use that one
                    next = prev;
                }
                // Advance tail
                Interlocked.CompareExchange(ref _tail, next, tail);
            }
            
            // Move to new segment
            tail = _tail;
            pos = Interlocked.Add(ref tail.EnqueuePos, 1) - 1;
        }
        
        int index = (int)((pos - tail.BasePosition) & _segmentSizeMask);
        
        // Claim the slot using CAS on sequence
        int spinCount = 0;
        while (true)
        {
            long seq = Volatile.Read(ref tail.Sequences[index]);
            long expectedSeq = pos - tail.BasePosition;
            
            // Slot is available if current sequence == expected (not yet claimed)
            if (seq == expectedSeq)
            {
                if (Interlocked.CompareExchange(
                    ref tail.Sequences[index], 
                    expectedSeq + _segmentSize + 1, 
                    seq) == seq)
                {
                    tail.Buffer[index].Value = item;
                    // Signal that write is complete
                    Volatile.Write(ref tail.Sequences[index], expectedSeq + _segmentSize);
                    return true;
                }
            }
            else
            {
                // Check if we need to move to next segment
                if (pos >= tail.BasePosition + _segmentSize && tail != _tail)
                {
                    return TryEnqueue(item); // Retry with new tail
                }
                
                if (spinCount++ > 1000)
                {
                    Thread.SpinWait(1);
                }
            }
        }
    }

    /// <summary>
    /// Enqueues an item, spinning until successful.
    /// </summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public void Enqueue(T item)
    {
        while (!TryEnqueue(item))
        {
            Thread.SpinWait(10);
        }
    }

    /// <summary>
    /// Tries to dequeue an item. Returns true if successful.
    /// </summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public bool TryDequeue(out T item)
    {
        var head = _head;
        long pos = Interlocked.Add(ref head.DequeuePos, 1) - 1;
        
        // Check if we need to move to next segment
        if (pos >= head.BasePosition + _segmentSize)
        {
            Segment next = head.Next;
            if (next == null)
            {
                Interlocked.Decrement(ref head.DequeuePos);
                // Check if there's actually a next segment now
                if (head.Next != null)
                {
                    Interlocked.CompareExchange(ref _head, head.Next, head);
                    return TryDequeue(out item);
                }
                item = default;
                return false;
            }
            
            Interlocked.CompareExchange(ref _head, next, head);
            head = next;
            pos = Interlocked.Add(ref head.DequeuePos, 1) - 1;
        }
        
        int index = (int)((pos - head.BasePosition) & _segmentSizeMask);
        int spinCount = 0;
        
        while (true)
        {
            long seq = Volatile.Read(ref head.Sequences[index]);
            long expectedSeq = pos - head.BasePosition;
            
            // Slot is ready when sequence indicates write complete (seq >= expected + segmentSize)
            if (seq >= expectedSeq + _segmentSize)
            {
                if (Interlocked.CompareExchange(
                    ref head.Sequences[index],
                    expectedSeq + 1,
                    seq) == seq)
                {
                    item = head.Buffer[index].Value;
                    head.Buffer[index].Value = default; // Help GC
                    return true;
                }
            }
            else
            {
                // Check if we need to move to next segment
                if (pos >= head.BasePosition + _segmentSize || head.DequeuePos > head.BasePosition + _segmentSize)
                {
                    Interlocked.Decrement(ref head.DequeuePos);
                    return TryDequeue(out item);
                }
                
                // Check if empty
                if (Volatile.Read(ref _tail.EnqueuePos) <= pos)
                {
                    if (spinCount++ > 1000)
                    {
                        Interlocked.Decrement(ref head.DequeuePos);
                        item = default;
                        return false;
                    }
                }
                
                if (spinCount > 10)
                    Thread.SpinWait(1);
            }
        }
    }

    /// <summary>
    /// Tries to peek at the next item without removing it.
    /// </summary>
    public bool TryPeek(out T item)
    {
        var head = _head;
        long pos = head.DequeuePos;
        
        if (pos >= head.BasePosition + _segmentSize)
        {
            Segment next = head.Next;
            if (next == null)
            {
                item = default;
                return false;
            }
            head = next;
            pos = head.DequeuePos;
        }
        
        int index = (int)((pos - head.BasePosition) & _segmentSizeMask);
        long seq = Volatile.Read(ref head.Sequences[index]);
        long expectedSeq = pos - head.BasePosition;
        
        if (seq >= expectedSeq + _segmentSize)
        {
            item = head.Buffer[index].Value;
            return true;
        }
        
        item = default;
        return false;
    }

    /// <summary>
    /// Dequeues an item, spinning until successful.
    /// </summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public T Dequeue()
    {
        T item;
        while (!TryDequeue(out item))
        {
            Thread.SpinWait(10);
        }
        return item;
    }

    /// <summary>
    /// Approximate count (may be stale immediately).
    /// </summary>
    public long Count
    {
        get
        {
            long totalEnqueued = 0;
            long totalDequeued = 0;
            
            // Get approximate positions from all segments
            Segment current = _head;
            while (current != null)
            {
                totalEnqueued = System.Math.Max(totalEnqueued, current.BasePosition + current.EnqueuePos);
                totalDequeued = System.Math.Max(totalDequeued, current.BasePosition + current.DequeuePos);
                current = current.Next;
            }
            
            return totalEnqueued - totalDequeued;
        }
    }

    /// <summary>
    /// Checks if queue is empty.
    /// </summary>
    public bool IsEmpty
    {
        get
        {
            var head = _head;
            return head.DequeuePos >= Volatile.Read(ref _tail.EnqueuePos) && head.Next == null;
        }
    }

    /// <summary>
    /// Clears all items from the queue.
    /// </summary>
    public void Clear()
    {
        while (TryDequeue(out _)) { }
    }

    /// <summary>
    /// A segment of the queue - fixed-size ring buffer.
    /// </summary>
    private sealed class Segment
    {
        public readonly Cell[] Buffer;
        public readonly long[] Sequences;
        public readonly long BasePosition;
        
        // Use separate atomic counters instead of Cell.Sequence array
        public long EnqueuePos;
        public long DequeuePos;
        
        public Segment? Next;
        
        public Segment(int size, long basePosition)
        {
            Buffer = new Cell[size];
            Sequences = new long[size];
            BasePosition = basePosition;
            
            // Initialize sequences to basePosition for all slots
            for (int i = 0; i < size; i++)
            {
                Sequences[i] = basePosition;
                Buffer[i] = new Cell();
            }
        }
        
        // Indexer for easier access
        public Cell this[int index] => Buffer[index];
        
        public struct Cell
        {
            public T Value;
        }
    }
}
