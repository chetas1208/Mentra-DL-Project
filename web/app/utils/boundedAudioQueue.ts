// Bounded outbound-frame queue with a drop-OLDEST policy (spec section
// 22, BACKPRESSURE). Used when the WebSocket's own send buffer
// (bufferedAmount) is already full: rather than growing an unbounded
// backlog of stale audio, or silently discarding without telemetry, this
// queue caps depth and reports exactly what it dropped.
//
// Drop-oldest, not drop-newest: for a live wearer-detection session the
// most RECENT audio is the one that matters for a live decision --
// catching up on a half-second-old backlog is worse than skipping it and
// resuming from "now".

export interface QueuedAudioFrame {
  sequenceNumber: number
  captureTimestampNs: bigint
  pcm16: Uint8Array
}

export interface DropEvent {
  framesDropped: number
  bytesDropped: number
}

export class BoundedAudioQueue {
  private items: QueuedAudioFrame[] = []
  private totalFramesDropped = 0
  private totalBytesDropped = 0
  private peakDepth = 0
  private readonly maxFrames: number

  constructor(maxFrames: number) {
    if (maxFrames <= 0) throw new Error('BoundedAudioQueue: maxFrames must be > 0')
    this.maxFrames = maxFrames
  }

  /** Enqueues one frame; if this pushes depth past maxFrames, drops the
   * OLDEST queued frame (not this new one) and returns the drop event
   * (null if nothing was dropped). */
  push(frame: QueuedAudioFrame): DropEvent | null {
    this.items.push(frame)
    this.peakDepth = Math.max(this.peakDepth, this.items.length)
    if (this.items.length > this.maxFrames) {
      const dropped = this.items.shift()!
      this.totalFramesDropped += 1
      this.totalBytesDropped += dropped.pcm16.length
      return { framesDropped: 1, bytesDropped: dropped.pcm16.length }
    }
    return null
  }

  shift(): QueuedAudioFrame | undefined {
    return this.items.shift()
  }

  get depth(): number {
    return this.items.length
  }

  get highWaterMark(): number {
    return this.peakDepth
  }

  get framesDropped(): number {
    return this.totalFramesDropped
  }

  get bytesDropped(): number {
    return this.totalBytesDropped
  }

  clear(): void {
    this.items = []
  }
}
