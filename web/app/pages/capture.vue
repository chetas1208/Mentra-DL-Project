<script setup lang="ts">
// G4 WS6 -- research capture mode.
//
// A deliberately plain operator console for pilot recording sessions. It is
// NOT a redesign of the live product page: it reuses useRuntimeState (device
// selection, mic capture, transport) unchanged and adds only the fields a
// research take needs. Keep it boring -- during a pilot the operator is
// managing a person wearing glasses, not admiring a UI.
//
// Real remaining blocker, stated in the UI itself so nobody misreads a take:
// this page works today with any input device; what makes a capture *Mentra
// evidence* is a physical Mentra Live paired to THIS machine and the operator
// honestly declaring it below.

import { SOURCE_KINDS, MENTRA_HARDWARE_SOURCES, type SourceKind } from '~/composables/useResearchCapture'

const {
  devices, selectedDeviceId, refreshDevices, selectDevice, attachDeviceChangeListener,
  mic, transport,
} = useRuntimeState()

const { isCapturing, captureFormat, micLevel, framesConverted, start: startMic, stop: stopMic } = mic
const { wsState, capabilities, framesSent, connect, disconnect } = transport
const capture = useResearchCapture()

const wearerId = ref(capture.newAnonymousWearerId())
const sessionId = ref(capture.newSessionId())
const condition = ref('')
const sourceKind = ref<SourceKind | ''>('')
const glassesModel = ref('')
const mentraosVersion = ref('')
const firmwareVersion = ref('')
const notes = ref('')
const errorMessage = ref<string | null>(null)

onMounted(() => {
  refreshDevices().catch(() => {})
  attachDeviceChangeListener()
})

const declaresMentraHardware = computed(
  () => !!sourceKind.value && MENTRA_HARDWARE_SOURCES.includes(sourceKind.value as SourceKind),
)
const canBegin = computed(
  () => isCapturing.value && wsState.value === 'CONNECTED'
    && !!condition.value.trim() && !!sourceKind.value && !capture.isRecording.value,
)

async function startAudio() {
  errorMessage.value = null
  try {
    if (wsState.value !== 'CONNECTED') await connect()
    await startMic(selectedDeviceId.value, (pcm16, ts) => transport.sendAudioFrame(pcm16, ts))
    await refreshDevices()
  } catch (e) {
    errorMessage.value = e instanceof Error ? e.message : String(e)
  }
}

function beginTake() {
  errorMessage.value = null
  const ok = capture.begin({
    sessionId: sessionId.value,
    wearerId: wearerId.value,
    condition: condition.value.trim(),
    sourceKind: sourceKind.value as SourceKind,
    glassesModel: glassesModel.value.trim(),
    mentraosVersion: mentraosVersion.value.trim(),
    firmwareVersion: firmwareVersion.value.trim(),
    notes: notes.value.trim(),
  })
  if (!ok) errorMessage.value = capture.captureError.value
}

/** Ending a take disconnects: the backend finalises, validates and writes the
 * session on transport close. That ordering is deliberate -- a take is only
 * "done" once the file exists on disk with its validation report. */
function endTake() {
  capture.end()
  stopMic()
  disconnect()
  sessionId.value = capture.newSessionId()
}

function newWearer() {
  wearerId.value = capture.newAnonymousWearerId()
}
</script>

<template>
  <main class="min-h-screen bg-panel-bg text-ink font-mono p-4 sm:p-8">
    <div class="w-full max-w-2xl mx-auto space-y-6">
      <header>
        <h1 class="font-display text-2xl font-semibold tracking-tight">Research capture</h1>
        <p class="text-ink-dim text-xs mt-1 tracking-wide">
          PILOT SESSION RECORDER · RAW NATIVE-RATE + DERIVED 16 kHz · STAYS LOCAL
        </p>
      </header>

      <section class="border border-ink-dim/30 rounded p-4 space-y-3">
        <h2 class="text-sm tracking-wide text-ink-dim">1 · AUDIO SOURCE</h2>
        <select
          v-model="selectedDeviceId"
          class="w-full bg-transparent border border-ink-dim/40 rounded px-2 py-1 text-sm"
          :disabled="isCapturing"
          @change="selectDevice(selectedDeviceId ?? undefined)"
        >
          <option :value="undefined">Default microphone</option>
          <option v-for="d in devices" :key="d.deviceId" :value="d.deviceId">
            {{ d.label || d.deviceId }}
          </option>
        </select>
        <div class="flex gap-2">
          <button
            class="border border-ink-dim/40 rounded px-3 py-1 text-sm"
            :disabled="isCapturing"
            @click="startAudio"
          >Start audio</button>
          <button
            class="border border-ink-dim/40 rounded px-3 py-1 text-sm"
            :disabled="!isCapturing"
            @click="endTake"
          >Stop &amp; finalise</button>
        </div>
        <dl class="text-xs text-ink-dim grid grid-cols-2 gap-x-4 gap-y-1">
          <dt>Backend</dt><dd>{{ wsState }}</dd>
          <dt>Model</dt><dd>{{ capabilities?.modelId ?? 'NOT AVAILABLE' }}</dd>
          <dt>Native rate</dt><dd>{{ captureFormat.sampleRate ?? '—' }} Hz</dd>
          <dt>Channels</dt><dd>{{ captureFormat.channelCount ?? '—' }}</dd>
          <dt>AGC granted</dt><dd>{{ captureFormat.autoGainControl ?? 'unknown' }}</dd>
          <dt>Noise suppression granted</dt><dd>{{ captureFormat.noiseSuppression ?? 'unknown' }}</dd>
          <dt>Echo cancellation granted</dt><dd>{{ captureFormat.echoCancellation ?? 'unknown' }}</dd>
          <dt>Mic level</dt><dd>{{ micLevel.toFixed(4) }}</dd>
        </dl>
        <p class="text-[11px] text-ink-dim">
          These are the settings the browser actually granted, read back from the track — not what was requested.
        </p>
      </section>

      <section class="border border-ink-dim/30 rounded p-4 space-y-3">
        <h2 class="text-sm tracking-wide text-ink-dim">2 · TAKE METADATA</h2>
        <label class="block text-xs">Session
          <input v-model="sessionId" class="w-full bg-transparent border border-ink-dim/40 rounded px-2 py-1 text-sm mt-1">
        </label>
        <label class="block text-xs">Anonymous wearer ID
          <div class="flex gap-2 mt-1">
            <input v-model="wearerId" class="flex-1 bg-transparent border border-ink-dim/40 rounded px-2 py-1 text-sm">
            <button class="border border-ink-dim/40 rounded px-3 text-sm" @click="newWearer">New</button>
          </div>
        </label>
        <label class="block text-xs">Condition <span class="text-wearer">(required)</span>
          <input
            v-model="condition"
            placeholder="e.g. quiet_room_solo / bystander_1m / machinery_loud"
            class="w-full bg-transparent border border-ink-dim/40 rounded px-2 py-1 text-sm mt-1"
          >
        </label>
        <label class="block text-xs">Source <span class="text-wearer">(required)</span>
          <select v-model="sourceKind" class="w-full bg-transparent border border-ink-dim/40 rounded px-2 py-1 text-sm mt-1">
            <option value="">— declare what this audio really is —</option>
            <option v-for="k in SOURCE_KINDS" :key="k" :value="k">{{ k }}</option>
          </select>
        </label>
        <template v-if="declaresMentraHardware">
          <label class="block text-xs">Glasses model
            <input v-model="glassesModel" class="w-full bg-transparent border border-ink-dim/40 rounded px-2 py-1 text-sm mt-1">
          </label>
          <label class="block text-xs">MentraOS version
            <input v-model="mentraosVersion" class="w-full bg-transparent border border-ink-dim/40 rounded px-2 py-1 text-sm mt-1">
          </label>
          <label class="block text-xs">Glasses firmware version
            <input v-model="firmwareVersion" class="w-full bg-transparent border border-ink-dim/40 rounded px-2 py-1 text-sm mt-1">
          </label>
        </template>
        <label class="block text-xs">Notes
          <textarea v-model="notes" rows="2" class="w-full bg-transparent border border-ink-dim/40 rounded px-2 py-1 text-sm mt-1" />
        </label>
        <p v-if="sourceKind && !declaresMentraHardware" class="text-[11px] text-ink-dim border border-ink-dim/30 rounded p-2">
          This take will be recorded as <strong>NOT Mentra hardware audio</strong> and must never be
          reported as a Mentra device result.
        </p>
      </section>

      <section class="border border-ink-dim/30 rounded p-4 space-y-3">
        <h2 class="text-sm tracking-wide text-ink-dim">3 · RECORD</h2>
        <div class="flex gap-2">
          <button
            class="border border-ink-dim/40 rounded px-3 py-1 text-sm"
            :disabled="!canBegin"
            @click="beginTake"
          >Begin take</button>
          <button
            class="border border-ink-dim/40 rounded px-3 py-1 text-sm"
            :disabled="!capture.isRecording.value"
            @click="endTake"
          >End take</button>
        </div>
        <dl class="text-xs text-ink-dim grid grid-cols-2 gap-x-4 gap-y-1">
          <dt>Recording</dt><dd>{{ capture.isRecording.value ? 'YES' : 'no' }}</dd>
          <dt>Raw seconds sent</dt><dd>{{ capture.rawSecondsSent.value.toFixed(1) }}</dd>
          <dt>Raw frames sent</dt><dd>{{ capture.rawFramesSent.value }}</dd>
          <dt>Raw frames dropped</dt><dd>{{ capture.rawFramesDropped.value }}</dd>
          <dt>16 kHz frames sent</dt><dd>{{ framesSent }} ({{ framesConverted }} converted)</dd>
          <dt>Backend session</dt>
          <dd>{{ (capture.captureAck.value as Record<string, unknown> | null)?.status ?? '—' }}</dd>
        </dl>
        <p v-if="errorMessage" class="text-xs text-environment">{{ errorMessage }}</p>
        <p class="text-[11px] text-ink-dim">
          The backend writes <code>raw.wav</code>, <code>derived_16k.wav</code>,
          <code>metadata.json</code> and <code>validation.json</code> when the transport session
          closes. Start the receiver with <code>--capture-dir</code>, otherwise capture messages
          are ignored and only live inference runs.
        </p>
      </section>
    </div>
  </main>
</template>
