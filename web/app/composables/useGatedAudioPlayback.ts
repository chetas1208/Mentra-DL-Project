// Schedules incoming PCM16 chunks for gapless playback -- one independent
// track each for WEARER_PCM and ENVIRONMENT_PCM. These are GATED audio
// (silence outside the matching classification state), not truly
// separated overlapping speech -- see docs/MENTRA_REMOTE_AUDIO.md.
//
// Uses back-to-back AudioBufferSourceNode scheduling (the standard
// low-complexity pattern for streaming PCM playback) rather than a custom
// playback AudioWorklet -- simpler to get right, and this isn't
// latency-critical the way capture is.

import { pcm16LEToFloat32 } from '~/utils/mentraProtocol'

const START_LOOKAHEAD_S = 0.08 // small jitter cushion before first chunk plays

function makeTrack(context: AudioContext, gainValue: number) {
  const gainNode = context.createGain()
  gainNode.gain.value = gainValue
  gainNode.connect(context.destination)
  let nextStartTime = 0

  function push(samples: Float32Array) {
    if (samples.length === 0) return
    const buffer = context.createBuffer(1, samples.length, 16000)
    buffer.copyToChannel(samples, 0)
    const source = context.createBufferSource()
    source.buffer = buffer
    source.connect(gainNode)

    const now = context.currentTime
    if (nextStartTime < now + START_LOOKAHEAD_S) {
      // fell behind (or first chunk) -- resync to now + cushion rather
      // than trying to catch up by scheduling in the past
      nextStartTime = now + START_LOOKAHEAD_S
    }
    source.start(nextStartTime)
    nextStartTime += buffer.duration
  }

  return { push, gainNode }
}

export type PlaybackMode = 'both' | 'wearer' | 'environment' | 'muted'

export function useGatedAudioPlayback() {
  const playbackMode = useState<PlaybackMode>('mentra-playback-mode', () => 'both')
  let context: AudioContext | null = null
  let wearerTrack: ReturnType<typeof makeTrack> | null = null
  let environmentTrack: ReturnType<typeof makeTrack> | null = null

  function ensureContext() {
    if (!context) {
      context = new AudioContext({ sampleRate: 16000 })
      wearerTrack = makeTrack(context, playbackMode.value === 'both' || playbackMode.value === 'wearer' ? 1 : 0)
      environmentTrack = makeTrack(context, playbackMode.value === 'both' || playbackMode.value === 'environment' ? 1 : 0)
    }
    return context
  }

  function pushWearerPcm(bytes: Uint8Array) {
    ensureContext()
    wearerTrack!.push(pcm16LEToFloat32(bytes))
  }

  function pushEnvironmentPcm(bytes: Uint8Array) {
    ensureContext()
    environmentTrack!.push(pcm16LEToFloat32(bytes))
  }

  /** "Host" = the enrolled wearer's gated track. "Surround" = everyone
   * else's gated track. Independent gain per track, so this is a real
   * per-source mute, not a single global on/off -- lets someone hear just
   * their own voice, just the room around them, both, or neither. */
  function setPlaybackMode(mode: PlaybackMode) {
    playbackMode.value = mode
    if (wearerTrack) wearerTrack.gainNode.gain.value = (mode === 'both' || mode === 'wearer') ? 1 : 0
    if (environmentTrack) environmentTrack.gainNode.gain.value = (mode === 'both' || mode === 'environment') ? 1 : 0
  }

  function close() {
    context?.close()
    context = null
    wearerTrack = null
    environmentTrack = null
  }

  return { playbackMode, pushWearerPcm, pushEnvironmentPcm, setPlaybackMode, close }
}
