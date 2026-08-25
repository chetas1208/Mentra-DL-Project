<script setup lang="ts">
// Never communicate status by color alone (section 48) -- every row pairs
// a symbol + short text + color.
type RowState = 'ok' | 'warn' | 'bad' | 'neutral'

const props = defineProps<{
  inputLabel: string
  inputState: RowState
  audioLabel: string
  audioState: RowState
  backendLabel: string
  backendState: RowState
  modelLabel: string
  modelState: RowState
}>()

const SYMBOL: Record<RowState, string> = { ok: '●', warn: '◐', bad: '✕', neutral: '○' }
const COLOR: Record<RowState, string> = {
  ok: 'text-wearer', warn: 'text-environment', bad: 'text-red-400', neutral: 'text-ink-faint',
}

const rows = computed(() => [
  { name: 'INPUT', label: props.inputLabel, state: props.inputState },
  { name: 'AUDIO', label: props.audioLabel, state: props.audioState },
  { name: 'BACKEND', label: props.backendLabel, state: props.backendState },
  { name: 'MODEL', label: props.modelLabel, state: props.modelState },
])
</script>

<template>
  <div class="grid grid-cols-2 sm:grid-cols-4 gap-3 text-[11px]">
    <div v-for="row in rows" :key="row.name" class="flex flex-col gap-0.5">
      <span class="text-ink-faint tracking-wide">{{ row.name }}</span>
      <span class="font-mono flex items-center gap-1.5" :class="COLOR[row.state]">
        <span aria-hidden="true">{{ SYMBOL[row.state] }}</span>
        <span>{{ row.label }}</span>
      </span>
    </div>
  </div>
</template>
