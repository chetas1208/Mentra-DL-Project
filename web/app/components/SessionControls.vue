<script setup lang="ts">
defineProps<{
  isCapturing: boolean
  isStarting: boolean
  disabled: boolean
  errorMessage: string | null
}>()
const emit = defineEmits<{ start: []; stop: [] }>()
</script>

<template>
  <div class="space-y-3">
    <button
      v-if="!isCapturing"
      :disabled="isStarting || disabled"
      class="w-full min-h-[44px] border border-wearer/40 text-wearer hover:bg-wearer hover:text-panel-bg focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-wearer focus-visible:ring-offset-2 focus-visible:ring-offset-panel-surface transition-colors rounded-md py-2.5 text-xs font-semibold tracking-wide disabled:opacity-50 disabled:cursor-not-allowed"
      @click="emit('start')"
    >
      {{ isStarting ? 'CONNECTING…' : 'START SESSION' }}
    </button>
    <button
      v-else
      class="w-full min-h-[44px] border border-red-500/40 text-red-400 hover:bg-red-500 hover:text-panel-bg focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-red-400 focus-visible:ring-offset-2 focus-visible:ring-offset-panel-surface transition-colors rounded-md py-2.5 text-xs font-semibold tracking-wide"
      @click="emit('stop')"
    >
      STOP SESSION
    </button>

    <p v-if="errorMessage" role="alert" class="text-xs text-red-400 bg-red-950/40 border border-red-900 rounded-md px-3 py-2 font-sans">
      {{ errorMessage }}
    </p>
  </div>
</template>
