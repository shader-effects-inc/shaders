/**
 * Performance tracking for shader rendering
 * Tracks frame times, memory usage, complexity, and more
 */

export interface PerformanceStats {
    // Frame timing
    fps: number
    avgFrameTime: number
    minFrameTime: number
    maxFrameTime: number
    p99FrameTime: number
    stdDevFrameTime: number

    // Jank detection
    jankCount: number
    jankPercent: number

    // Complexity
    nodeCount: number
    rttNodeCount: number
    complexityScore: number

    // Memory (Chrome only)
    memoryUsedMB: number | null
    memoryGrowthRate: number | null

    // CPU/GPU breakdown
    cpuTime: number | null
    gpuTime: number | null
    budgetUsed: number

    // Renderer info
    drawCalls: number | null
    shaderPrograms: number | null
    textureCount: number | null

    // Intensity score (normalized 0-100)
    intensityScore: number
    intensityLabel: string

    // State
    isRendering: boolean
}

export class PerformanceTracker {
    private frameTimesMs: number[] = []
    private frameTimesIndex = 0 // Circular buffer write index
    private frameTimesCount = 0 // Number of valid entries
    private readonly maxSamples = 60
    private readonly targetFrameTime = 16.67 // 60 FPS budget

    private jankFrameCount = 0
    private totalFrameCount = 0

    private memorySnapshots: { time: number, bytes: number }[] = []
    private memorySnapshotIndex = 0 // Circular buffer write index
    private readonly maxMemorySnapshots = 6 // ~5 seconds at one sample a second
    /** Reading the heap size isn't free: one sample a second is plenty for a ~5s growth rate. */
    private readonly memorySnapshotIntervalMs = 1000
    private lastMemorySnapshotTime = -Infinity

    private nodeCount = 0
    private rttNodeCount = 0

    private lastCpuTime: number | null = null
    private lastGpuTime: number | null = null

    private isRendering = false

    // Track actual frame intervals for accurate FPS
    private lastFrameTimestamp = 0
    private frameIntervals: number[] = []
    private frameIntervalsIndex = 0 // Circular buffer write index
    private frameIntervalsCount = 0 // Number of valid entries

    /**
     * Adds a value to a circular buffer, returning the active slice for calculations
     */
    private pushToCircularBuffer(
        buffer: number[],
        value: number,
        index: number,
        _count: number,
        maxSize: number
    ): { index: number, count: number } {
        if (buffer.length < maxSize) {
            buffer.push(value)
            return { index: buffer.length, count: buffer.length }
        }
        buffer[index % maxSize] = value
        return { index: index + 1, count: maxSize }
    }

    /**
     * Gets the active values from a circular buffer
     */
    private getBufferValues(buffer: number[], count: number): number[] {
        return count <= buffer.length ? buffer.slice(0, count) : buffer
    }

    /**
     * Records a frame's render time
     */
    recordFrame(frameTimeMs: number): void {
        const result = this.pushToCircularBuffer(
            this.frameTimesMs, frameTimeMs,
            this.frameTimesIndex, this.frameTimesCount, this.maxSamples
        )
        this.frameTimesIndex = result.index
        this.frameTimesCount = result.count

        this.totalFrameCount++
        if (frameTimeMs > this.targetFrameTime) {
            this.jankFrameCount++
        }

        // Track actual frame intervals for accurate FPS calculation
        const now = performance.now()
        if (this.lastFrameTimestamp > 0) {
            const interval = now - this.lastFrameTimestamp
            const intervalResult = this.pushToCircularBuffer(
                this.frameIntervals, interval,
                this.frameIntervalsIndex, this.frameIntervalsCount, this.maxSamples
            )
            this.frameIntervalsIndex = intervalResult.index
            this.frameIntervalsCount = intervalResult.count
        }
        this.lastFrameTimestamp = now

        // Record memory snapshot (at most once a second)
        this.recordMemorySnapshot(now)
    }

    /**
     * Records CPU time for the last frame
     */
    recordCpuTime(timeMs: number): void {
        this.lastCpuTime = timeMs
    }

    /**
     * Records GPU time for the last frame
     */
    recordGpuTime(timeMs: number): void {
        this.lastGpuTime = timeMs
    }

    /**
     * Updates node counts for complexity calculation
     */
    updateNodeCounts(nodeCount: number, rttNodeCount: number): void {
        this.nodeCount = nodeCount
        this.rttNodeCount = rttNodeCount
    }

    /**
     * Sets rendering state
     */
    setRendering(rendering: boolean): void {
        this.isRendering = rendering
    }

    /**
     * Records a memory snapshot for growth rate calculation
     * Uses circular buffer to avoid GC pressure from allocations
     */
    private recordMemorySnapshot(now: number): void {
        if (now - this.lastMemorySnapshotTime < this.memorySnapshotIntervalMs) return
        this.lastMemorySnapshotTime = now
        if (!(performance as any).memory) return

        const bytes = (performance as any).memory.usedJSHeapSize

        // Use circular buffer - reuse existing objects, no allocations after warmup
        if (this.memorySnapshots.length < this.maxMemorySnapshots) {
            // Still filling up - allocate new object
            this.memorySnapshots.push({ time: now, bytes })
        } else {
            // Buffer full - overwrite oldest entry in-place (no allocation)
            const entry = this.memorySnapshots[this.memorySnapshotIndex]
            entry.time = now
            entry.bytes = bytes
            this.memorySnapshotIndex = (this.memorySnapshotIndex + 1) % this.maxMemorySnapshots
        }
    }

    /**
     * Calculates memory growth rate in MB/sec
     * Works with circular buffer - oldest is at write index, newest is before it
     */
    private calculateMemoryGrowthRate(): number | null {
        if (this.memorySnapshots.length < 2) return null

        // In circular buffer: oldest is at current write index, newest is one before
        const len = this.memorySnapshots.length
        const oldestIdx = len < this.maxMemorySnapshots ? 0 : this.memorySnapshotIndex
        const newestIdx = len < this.maxMemorySnapshots ? len - 1 : (this.memorySnapshotIndex - 1 + len) % len

        const oldest = this.memorySnapshots[oldestIdx]
        const newest = this.memorySnapshots[newestIdx]

        const timeDeltaSeconds = (newest.time - oldest.time) / 1000
        if (timeDeltaSeconds <= 0) return null

        const bytesDelta = newest.bytes - oldest.bytes
        const mbDelta = bytesDelta / (1024 * 1024)

        return mbDelta / timeDeltaSeconds
    }

    /**
     * Calculates standard deviation of frame times
     * Optimized to avoid array allocations
     */
    private calculateStdDev(values: number[]): number {
        if (values.length === 0) return 0

        const mean = values.reduce((a, b) => a + b, 0) / values.length
        let sumSquaredDiff = 0
        for (let i = 0; i < values.length; i++) {
            const diff = values[i] - mean
            sumSquaredDiff += diff * diff
        }

        return Math.sqrt(sumSquaredDiff / values.length)
    }

    // Pre-allocated sort buffer to avoid allocations in calculateP99
    private sortBuffer: number[] = []

    /**
     * Calculates 99th percentile frame time
     * Uses pre-allocated buffer and in-place sorting to avoid allocations
     */
    private calculateP99(values: number[]): number {
        const len = values.length
        if (len === 0) return 0

        // Ensure buffer is large enough
        while (this.sortBuffer.length < len) {
            this.sortBuffer.push(0)
        }

        // Copy values to buffer
        for (let i = 0; i < len; i++) {
            this.sortBuffer[i] = values[i]
        }

        // In-place insertion sort on just the portion we need (small arrays, avoids allocation)
        for (let i = 1; i < len; i++) {
            const current = this.sortBuffer[i]
            let j = i - 1
            while (j >= 0 && this.sortBuffer[j] > current) {
                this.sortBuffer[j + 1] = this.sortBuffer[j]
                j--
            }
            this.sortBuffer[j + 1] = current
        }

        const index = Math.floor(len * 0.99)
        return this.sortBuffer[Math.min(index, len - 1)]
    }

    /**
     * Calculates complexity score based on node structure
     * RTT nodes are 10x more expensive than regular nodes
     */
    private calculateComplexityScore(): number {
        return this.nodeCount + (this.rttNodeCount * 10)
    }

    /**
     * Calculates normalized intensity score (0-100) for "average machine" estimation
     * Combines frame time, complexity, and GPU time into single metric
     *
     * Scoring:
     * - 0-20: Very Light (simple effects, <5ms)
     * - 21-40: Light (basic effects, 5-8ms)
     * - 41-60: Medium (moderate effects, 8-12ms)
     * - 61-80: Heavy (complex effects, 12-15ms)
     * - 81-100: Very Heavy (intensive effects, >15ms)
     */
    private calculateIntensityScore(): { score: number, label: string } {
        const frameTimes = this.getBufferValues(this.frameTimesMs, this.frameTimesCount)
        const frameCount = frameTimes.length
        if (frameCount === 0) {
            return { score: 0, label: 'N/A' }
        }

        // Get average frame time
        const avgFrameTime = frameTimes.reduce((a, b) => a + b, 0) / frameCount

        // Calculate component scores (each 0-100)
        // Frame time score: maps 0-16.67ms to 0-100
        const frameTimeScore = Math.min((avgFrameTime / 16.67) * 100, 100)

        // Complexity score: maps based on complexity thresholds
        const complexity = this.calculateComplexityScore()
        const complexityScore = Math.min((complexity / 100) * 100, 100)

        // GPU time score (if available): maps 0-16.67ms to 0-100
        let gpuTimeScore = 0
        if (this.lastGpuTime !== null) {
            gpuTimeScore = Math.min((this.lastGpuTime / 16.67) * 100, 100)
        }

        // Weighted average (frame time is most important, then GPU, then complexity)
        const weights = this.lastGpuTime !== null
            ? { frame: 0.4, gpu: 0.4, complexity: 0.2 }
            : { frame: 0.7, gpu: 0, complexity: 0.3 }

        const finalScore = Math.round(
            frameTimeScore * weights.frame +
            gpuTimeScore * weights.gpu +
            complexityScore * weights.complexity
        )

        // Determine label
        let label: string
        if (finalScore <= 20) label = 'Very Light'
        else if (finalScore <= 40) label = 'Light'
        else if (finalScore <= 60) label = 'Medium'
        else if (finalScore <= 80) label = 'Heavy'
        else label = 'Very Heavy'

        return { score: Math.min(finalScore, 100), label }
    }

    /**
     * Gets current performance statistics
     */
    getStats(rendererInfo?: any): PerformanceStats {
        const frameTimes = this.getBufferValues(this.frameTimesMs, this.frameTimesCount)
        const intervals = this.getBufferValues(this.frameIntervals, this.frameIntervalsCount)
        const frameCount = frameTimes.length

        // Frame timing - calculate FPS from actual frame intervals (not frame time)
        // This gives accurate FPS tied to RAF calls, capped by vsync
        const fps = intervals.length > 0
            ? 1000 / (intervals.reduce((a, b) => a + b, 0) / intervals.length)
            : 0
        const avgFrameTime = frameCount > 0 ? frameTimes.reduce((a, b) => a + b, 0) / frameCount : 0
        const minFrameTime = frameCount > 0 ? Math.min(...frameTimes) : 0
        const maxFrameTime = frameCount > 0 ? Math.max(...frameTimes) : 0
        const p99FrameTime = this.calculateP99(frameTimes)
        const stdDevFrameTime = this.calculateStdDev(frameTimes)

        // Jank
        const jankPercent = this.totalFrameCount > 0 ? (this.jankFrameCount / this.totalFrameCount) * 100 : 0

        // Memory
        const memory = (performance as any).memory
        const memoryUsedMB = memory ? memory.usedJSHeapSize / (1024 * 1024) : null
        const memoryGrowthRate = this.calculateMemoryGrowthRate()

        // Budget utilization
        const budgetUsed = avgFrameTime > 0 ? (avgFrameTime / this.targetFrameTime) * 100 : 0

        // Renderer info
        const drawCalls = rendererInfo?.render?.calls ?? null
        const shaderPrograms = rendererInfo?.programs?.length ?? null
        const textureCount = rendererInfo?.memory?.textures ?? null

        // Calculate intensity score
        const intensity = this.calculateIntensityScore()

        return {
            fps: Math.round(fps),
            avgFrameTime: Math.round(avgFrameTime * 100) / 100,
            minFrameTime: Math.round(minFrameTime * 100) / 100,
            maxFrameTime: Math.round(maxFrameTime * 100) / 100,
            p99FrameTime: Math.round(p99FrameTime * 100) / 100,
            stdDevFrameTime: Math.round(stdDevFrameTime * 100) / 100,
            jankCount: this.jankFrameCount,
            jankPercent: Math.round(jankPercent * 10) / 10,
            nodeCount: this.nodeCount,
            rttNodeCount: this.rttNodeCount,
            complexityScore: this.calculateComplexityScore(),
            memoryUsedMB: memoryUsedMB !== null ? Math.round(memoryUsedMB * 100) / 100 : null,
            memoryGrowthRate: memoryGrowthRate !== null ? Math.round(memoryGrowthRate * 1000) / 1000 : null,
            cpuTime: this.lastCpuTime !== null ? Math.round(this.lastCpuTime * 100) / 100 : null,
            gpuTime: this.lastGpuTime !== null ? Math.round(this.lastGpuTime * 100) / 100 : null,
            budgetUsed: Math.round(budgetUsed * 10) / 10,
            drawCalls,
            shaderPrograms,
            textureCount,
            intensityScore: intensity.score,
            intensityLabel: intensity.label,
            isRendering: this.isRendering
        }
    }

    /**
     * Resets all statistics
     */
    reset(): void {
        this.frameTimesMs.length = 0
        this.frameTimesIndex = 0
        this.frameTimesCount = 0
        this.frameIntervals.length = 0
        this.frameIntervalsIndex = 0
        this.frameIntervalsCount = 0
        this.lastFrameTimestamp = 0
        this.jankFrameCount = 0
        this.totalFrameCount = 0
        this.memorySnapshots.length = 0
        this.memorySnapshotIndex = 0
        this.lastMemorySnapshotTime = -Infinity
        this.sortBuffer.length = 0
        this.lastCpuTime = null
        this.lastGpuTime = null
    }
}
