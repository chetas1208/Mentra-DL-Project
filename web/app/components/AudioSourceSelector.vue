<script setup lang="ts">
import type { AudioInputOption } from '~/composables/useMediaDevices'

const props = defineProps<{
  devices: AudioInputOption[]
  selectedDeviceId: string | undefined
  disabled: boolean
  sourceLost: boolean
  sourceClassification: string
}>()
const emit = defineEmits<{ select: [deviceId: string | undefined]; 'select-another': [] }>()

function onChange(e: Event) {
  const value = (e.target as HTMLSelectElement).value
  emit('select', value === '' ? undefined : value)
}

const classificationClass = computed(() => {
  if (props.sourceClassification === 'MENTRA LIVE') return 'text-wearer'
  if (props.sourceClassification === 'BROWSER MICROPHONE') return 'text-ink-dim'
  return 'text-ink-faint'
})
</script>

<template>
  <div class="space-y-2">
    <div class="flex items-center justify-between gap-3">
      <label for="input-device" class="text-[11px] text-ink-dim tracking-wide shrink-0">INPUT</label>
      <div class="flex items-center gap-2 max-w-[70%]">
        <span class="text-[10px] font-mono shrink-0" :class="classificationClass">{{ sourceClassification }}</span>
        <select
          id="input-device"
          :value="selectedDeviceId ?? ''"
          :disabled="disabled"
          class="bg-panel-bg border border-panel-line rounded-md px-2.5 py-2 text-xs text-ink w-full focus:outline-none focus-visible:ring-2 focus-visible:ring-wearer focus-visible:ring-offset-2 focus-visible:ring-offset-panel-surface disabled:opacity-50"
          @change="onChange"
        >
          <option value="">Default microphone</option>
          <option v-for="d in devices" :key="d.deviceId" :value="d.deviceId">
            {{ d.label || 'UNKNOWN AUDIO INPUT' }}
          </option>
        </select>
      </div>
    </div>

    <div v-if="sourceLost" role="alert" class="flex items-center justify-between gap-3 text-xs text-red-400 bg-red-950/40 border border-red-900 rounded-md px-3 py-2 font-sans">
      <span>⚠ INPUT LOST — the selected device disappeared.</span>
      <button
        type="button"
        class="underline decoration-dotted underline-offset-2 shrink-0 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-red-400 rounded-sm"
        @click="emit('select-another')"
      >
        Select another input
      </button>
    </div>
  </div>
</template>
