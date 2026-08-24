<script setup lang="ts">
interface InputDevice {
  deviceId: string
  label: string
}

const permissionState = ref<'idle' | 'requesting' | 'granted' | 'denied'>('idle')
const errorMessage = ref<string | null>(null)
const devices = ref<InputDevice[]>([])
const selectedDeviceId = ref<string | null>(null)
const trackSettings = ref<Record<string, unknown> | null>(null)
let activeStream: MediaStream | null = null

const mentraMatches = computed(() =>
  devices.value.filter((d) => /mentra/i.test(d.label)),
)

async function requestPermissionAndList() {
  errorMessage.value = null
  permissionState.value = 'requesting'
  try {
    // A bare getUserMedia() call is what actually triggers the OS
    // permission prompt and populates real device *labels* -- without it,
    // enumerateDevices() returns anonymized IDs with empty labels.
    const probeStream = await navigator.mediaDevices.getUserMedia({ audio: true })
    probeStream.getTracks().forEach((t) => t.stop())
    permissionState.value = 'granted'
    await refreshDeviceList()
  } catch (e) {
    permissionState.value = 'denied'
    errorMessage.value = e instanceof Error ? e.message : String(e)
  }
}

async function refreshDeviceList() {
  const all = await navigator.mediaDevices.enumerateDevices()
  devices.value = all
    .filter((d) => d.kind === 'audioinput')
    .map((d) => ({ deviceId: d.deviceId, label: d.label || '(unlabeled input)' }))
}

async function inspectDevice(deviceId: string) {
  errorMessage.value = null
  trackSettings.value = null
  selectedDeviceId.value = deviceId
  stopActiveStream()
  try {
    const stream = await navigator.mediaDevices.getUserMedia({
      audio: {
        deviceId: { exact: deviceId },
        channelCount: 1,
        echoCancellation: false,
        noiseSuppression: false,
        autoGainControl: false,
      },
    })
    activeStream = stream
    // The real, measured settings the browser actually granted -- never
    // assume a device delivers 16kHz/mono just because that's what this
    // pipeline wants; this is the whole point of the page.
    trackSettings.value = stream.getAudioTracks()[0]?.getSettings() ?? null
  } catch (e) {
    errorMessage.value = e instanceof Error ? e.message : String(e)
  }
}

function stopActiveStream() {
  activeStream?.getTracks().forEach((t) => t.stop())
  activeStream = null
}

onBeforeUnmount(() => stopActiveStream())

onMounted(() => {
  navigator.mediaDevices?.addEventListener?.('devicechange', () => {
    if (permissionState.value === 'granted') refreshDeviceList()
  })
})

// --- Bluetooth device scan -------------------------------------------
// Real Web Bluetooth API, not a simulation. Two honest limitations up
// front: (1) Chrome/Edge desktop + Chrome Android only -- no Safari, no
// Firefox, at all, ever (not a "not yet", the API isn't planned there).
// (2) requestDevice() shows nearby BLE devices currently ADVERTISING, via
// the browser's own native picker -- it does NOT read the OS-level
// Bluetooth pairing list (Web Bluetooth deliberately can't see that, for
// privacy), and selecting a device here does not grant this page any
// audio-streaming capability -- Mentra's mic_pcm/mic_lc3 stream is only
// exposed through the native SDK (Android/iOS/React Native), not Web
// Bluetooth GATT. This is a discovery/identification check, nothing more.
interface BtFound {
  name: string
  id: string
  isMentra: boolean
  connectedGatt: boolean
}

const btSupported = ref(typeof navigator !== 'undefined' && 'bluetooth' in navigator)
const btState = ref<'idle' | 'scanning' | 'done' | 'denied' | 'unsupported'>('idle')
const btError = ref<string | null>(null)
const btFound = ref<BtFound[]>([])
const btKnownDevices = ref<BtFound[]>([]) // previously-granted, via getDevices() -- no picker needed

async function scanForBluetoothDevices() {
  btError.value = null
  if (!btSupported.value) {
    btState.value = 'unsupported'
    return
  }
  btState.value = 'scanning'
  try {
    // acceptAllDevices -- we don't know Mentra's advertised service UUIDs,
    // so a namePrefix filter would risk silently missing it. This is why
    // the browser's picker is unavoidable here (Web Bluetooth requires a
    // user gesture + an explicit device choice, by design -- there is no
    // silent "list everything nearby" API).
    const device = await navigator.bluetooth.requestDevice({ acceptAllDevices: true })
    const name = device.name || '(unnamed device)'
    const entry: BtFound = {
      name, id: device.id, isMentra: /mentra/i.test(name),
      connectedGatt: device.gatt?.connected ?? false,
    }
    // De-dupe by id across repeated scans.
    btFound.value = [entry, ...btFound.value.filter((d) => d.id !== entry.id)]
    btState.value = 'done'
  } catch (e) {
    // User closing the picker without choosing throws -- not a real error.
    if (e instanceof Error && e.name === 'NotFoundError') {
      btState.value = 'done'
    } else {
      btState.value = 'denied'
      btError.value = e instanceof Error ? e.message : String(e)
    }
  }
}

async function loadKnownBluetoothDevices() {
  if (!btSupported.value || !navigator.bluetooth.getDevices) return
  try {
    const known = await navigator.bluetooth.getDevices()
    btKnownDevices.value = known.map((d) => ({
      name: d.name || '(unnamed device)', id: d.id,
      isMentra: /mentra/i.test(d.name || ''),
      connectedGatt: d.gatt?.connected ?? false,
    }))
  } catch {
    // getDevices() is still gated behind a flag in some Chrome versions --
    // fail silently, the scan button still works regardless.
  }
}

onMounted(() => {
  loadKnownBluetoothDevices()
})
</script>

<template>
  <main class="min-h-screen overflow-x-hidden bg-panel-bg text-ink font-mono flex items-center justify-center p-4 sm:p-8 motion-reduce:[&_*]:!transition-none motion-reduce:[&_*]:!animate-none">
    <div class="w-full max-w-2xl">
      <div class="flex flex-wrap items-end justify-between gap-x-4 gap-y-2 mb-6 px-1">
        <div>
          <h1 class="font-display text-xl sm:text-2xl font-semibold tracking-tight text-ink">Audio Input Debug</h1>
          <p class="text-ink-dim text-xs mt-1 tracking-wide">RESOLVES: DOES THE OS EXPOSE MENTRA AS A MICROPHONE?</p>
        </div>
        <NuxtLink to="/" class="text-[10px] text-ink-faint hover:text-ink-dim underline decoration-dotted underline-offset-2 transition-colors focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-wearer rounded-sm shrink-0">
          ← BACK TO DASHBOARD
        </NuxtLink>
      </div>

      <p class="mb-4 px-1 text-[11px] text-ink-dim font-sans">
        Pair Mentra Live via Bluetooth, then request permission below.
      </p>

      <div class="instrument-panel-enter border border-panel-line rounded-xl bg-panel-surface overflow-hidden">
        <div class="px-6 sm:px-10 py-6 border-b border-panel-line">
          <button
            v-if="permissionState !== 'granted'"
            :disabled="permissionState === 'requesting'"
            class="w-full min-h-[44px] border border-wearer/40 text-wearer hover:bg-wearer hover:text-panel-bg focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-wearer focus-visible:ring-offset-2 focus-visible:ring-offset-panel-surface transition-colors rounded-md py-2.5 text-xs font-semibold tracking-wide disabled:opacity-50 disabled:cursor-not-allowed"
            @click="requestPermissionAndList"
          >
            {{ permissionState === 'requesting' ? 'REQUESTING…' : 'REQUEST MICROPHONE PERMISSION' }}
          </button>

          <div v-else class="flex items-center justify-between text-[11px]">
            <span class="text-ink-dim tracking-wide">PERMISSION</span>
            <span class="text-wearer font-mono">GRANTED</span>
          </div>

          <p v-if="errorMessage" role="alert" class="text-xs text-red-400 bg-red-950/40 border border-red-900 rounded-md px-3 py-2 font-sans mt-3">
            {{ errorMessage }}
          </p>
        </div>

        <template v-if="permissionState === 'granted'">
          <!-- headline verdict -->
          <div
            class="px-6 sm:px-10 py-4 border-b border-panel-line text-[11px] font-sans leading-relaxed"
            :class="mentraMatches.length ? 'bg-wearer/10 text-wearer' : 'bg-environment/10 text-environment'"
            role="status" aria-live="polite"
          >
            <span class="font-mono font-semibold tracking-wide">
              {{ mentraMatches.length ? `✓ FOUND ${mentraMatches.length} MATCHING INPUT${mentraMatches.length > 1 ? 'S' : ''}` : '✗ NO "MENTRA" INPUT FOUND' }}
            </span>
          </div>

          <div class="px-6 sm:px-10 py-5 space-y-2">
            <div class="text-[11px] text-ink-dim tracking-wide mb-2">
              AUDIO INPUTS ({{ devices.length }})
            </div>
            <p v-if="!devices.length" class="text-xs text-ink-faint font-sans">No audio input devices reported.</p>
            <button
              v-for="d in devices"
              :key="d.deviceId"
              type="button"
              class="w-full text-left min-h-[44px] px-3 py-2.5 rounded-md border transition-colors focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-wearer flex items-center justify-between gap-3"
              :class="[
                selectedDeviceId === d.deviceId ? 'border-wearer bg-wearer/10' : 'border-panel-line hover:border-ink-dim',
                /mentra/i.test(d.label) ? 'ring-1 ring-wearer/50' : '',
              ]"
              @click="inspectDevice(d.deviceId)"
            >
              <span class="text-xs" :class="/mentra/i.test(d.label) ? 'text-wearer font-semibold' : 'text-ink'">
                {{ d.label }}
              </span>
              <span v-if="/mentra/i.test(d.label)" class="text-[10px] font-mono text-wearer shrink-0">MENTRA ✓</span>
            </button>
          </div>

          <div v-if="trackSettings" class="px-6 sm:px-10 py-4 border-t border-panel-line">
            <div class="text-[11px] text-ink-dim tracking-wide mb-2">REPORTED TRACK SETTINGS</div>
            <div class="grid grid-cols-2 sm:grid-cols-3 gap-x-4 gap-y-2 text-xs font-mono">
              <div v-for="(value, key) in trackSettings" :key="key">
                <div class="text-[10px] text-ink-faint">{{ key }}</div>
                <div class="text-ink">{{ value }}</div>
              </div>
            </div>
          </div>
        </template>
      </div>

      <!-- Bluetooth device scan -->
      <div class="instrument-panel-enter instrument-panel-enter-delayed mt-6 border border-panel-line rounded-xl bg-panel-surface overflow-hidden">
        <div class="px-6 sm:px-10 py-6 border-b border-panel-line">
          <h2 class="font-display text-sm font-semibold text-ink mb-1">Bluetooth Device Scan</h2>
          <p class="text-[11px] text-ink-dim font-sans leading-relaxed mb-4">
            Real Web Bluetooth API — Chrome/Edge desktop and Chrome Android only, never Safari
            or Firefox. Shows nearby BLE devices currently advertising, via the browser's own
            picker. This identifies whether Mentra Live is discoverable — it does not grant any
            audio-streaming access (that only exists through Mentra's native SDK, see the
            dashboard's disclosure banner).
          </p>

          <p v-if="!btSupported" class="text-xs text-environment bg-environment/10 border border-environment/30 rounded-md px-3 py-2 font-sans">
            Web Bluetooth isn't available in this browser. Try Chrome or Edge on desktop or Android.
          </p>
          <button
            v-else
            :disabled="btState === 'scanning'"
            class="w-full min-h-[44px] border border-wearer/40 text-wearer hover:bg-wearer hover:text-panel-bg focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-wearer focus-visible:ring-offset-2 focus-visible:ring-offset-panel-surface transition-colors rounded-md py-2.5 text-xs font-semibold tracking-wide disabled:opacity-50 disabled:cursor-not-allowed"
            @click="scanForBluetoothDevices"
          >
            {{ btState === 'scanning' ? 'SCANNING…' : 'SCAN FOR BLUETOOTH DEVICES' }}
          </button>

          <p v-if="btError" role="alert" class="text-xs text-red-400 bg-red-950/40 border border-red-900 rounded-md px-3 py-2 font-sans mt-3">
            {{ btError }}
          </p>
        </div>

        <div v-if="btKnownDevices.length" class="px-6 sm:px-10 py-4 border-b border-panel-line">
          <div class="text-[11px] text-ink-dim tracking-wide mb-2">PREVIOUSLY GRANTED ({{ btKnownDevices.length }})</div>
          <div
            v-for="d in btKnownDevices" :key="'known-' + d.id"
            class="flex flex-wrap items-center justify-between gap-x-3 gap-y-1 px-3 py-2.5 rounded-md border mb-1.5 transition-colors"
            :class="d.isMentra ? 'border-wearer/50 bg-wearer/10' : 'border-panel-line'"
          >
            <span class="text-xs" :class="d.isMentra ? 'text-wearer font-semibold' : 'text-ink'">{{ d.name }}</span>
            <div class="flex items-center gap-2 shrink-0">
              <span v-if="d.isMentra" class="text-[10px] font-mono text-wearer">MENTRA ✓</span>
              <span class="text-[10px] font-mono" :class="d.connectedGatt ? 'text-wearer' : 'text-ink-faint'">
                {{ d.connectedGatt ? 'CONNECTED' : 'NOT CONNECTED' }}
              </span>
            </div>
          </div>
        </div>

        <div v-if="btFound.length" class="px-6 sm:px-10 py-4">
          <div class="text-[11px] text-ink-dim tracking-wide mb-2">SCAN RESULTS ({{ btFound.length }})</div>
          <div
            v-for="d in btFound" :key="'found-' + d.id"
            class="flex flex-wrap items-center justify-between gap-x-3 gap-y-1 px-3 py-2.5 rounded-md border mb-1.5 transition-colors"
            :class="d.isMentra ? 'border-wearer/50 bg-wearer/10' : 'border-panel-line'"
          >
            <span class="text-xs" :class="d.isMentra ? 'text-wearer font-semibold' : 'text-ink'">{{ d.name }}</span>
            <span v-if="d.isMentra" class="text-[10px] font-mono text-wearer shrink-0">MENTRA ✓</span>
          </div>
        </div>
      </div>
    </div>
  </main>
</template>

<style scoped>
.instrument-panel-enter {
  animation: panel-settle 0.5s cubic-bezier(0.16, 1, 0.3, 1) both;
}
.instrument-panel-enter-delayed {
  animation-delay: 0.08s;
}
@keyframes panel-settle {
  from {
    opacity: 0;
    transform: translateY(8px);
  }
  to {
    opacity: 1;
    transform: translateY(0);
  }
}
</style>
