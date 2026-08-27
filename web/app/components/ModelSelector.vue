<script setup lang="ts">
// One compact runtime-model selector, deliberately built to the exact
// structure and styling of AudioSourceSelector.vue -- same label treatment,
// same select classes, same focus-ring convention, same error block. The only
// intentionally new thing on this page is the control itself.
//
// Every value shown here comes from the backend's catalog. Nothing is
// hardcoded: readiness, the experimental flag and the unavailable reason are
// all reported by the receiver, never inferred from the model id.
import type { CatalogModel } from '~/utils/modelCatalog'

const props = defineProps<{
  models: CatalogModel[]
  selectedModel: string | null
  activeModel: string | null
  /** True while a session is live -- a model switch requires stopping first,
   * so the control is disabled rather than silently hot-swapping mid-stream. */
  disabled: boolean
  error: string | null
}>()
const emit = defineEmits<{ select: [modelId: string] }>()

function onChange(e: Event) {
  emit('select', (e.target as HTMLSelectElement).value)
}

const selected = computed(() =>
  props.models.find((m) => m.id === props.selectedModel) ?? null,
)

// Amber = "not yet fully validated", the same colour-for-meaning this page
// already uses for the PLACEHOLDER enrollment state. Paired with the literal
// word so the state never depends on colour alone.
const badge = computed(() => {
  if (!selected.value) return null
  if (!selected.value.ready) return { text: 'UNAVAILABLE', class: 'text-red-400' }
  if (selected.value.experimental) return { text: 'EXPERIMENTAL', class: 'text-amber-400' }
  return null
})

const message = computed(() => {
  if (props.error) return props.error
  if (selected.value && !selected.value.ready) {
    return selected.value.unavailableReason
      ? `${selected.value.displayName} failed to load.`
      : `${selected.value.displayName} is unavailable.`
  }
  return null
})
</script>

<template>
  <div class="space-y-2">
    <div class="flex items-center justify-between gap-3">
      <label for="runtime-model" class="text-[11px] text-ink-dim tracking-wide shrink-0">MODEL</label>
      <div class="flex items-center gap-2 max-w-[70%]">
        <span v-if="badge" class="text-[10px] font-mono shrink-0" :class="badge.class">{{ badge.text }}</span>
        <select
          id="runtime-model"
          :value="selectedModel ?? ''"
          :disabled="disabled || models.length < 2"
          class="bg-panel-bg border border-panel-line rounded-md px-2.5 py-2 text-xs text-ink w-full focus:outline-none focus-visible:ring-2 focus-visible:ring-wearer focus-visible:ring-offset-2 focus-visible:ring-offset-panel-surface disabled:opacity-50"
          @change="onChange"
        >
          <option v-if="models.length === 0" value="">NOT AVAILABLE</option>
          <option v-for="m in models" :key="m.id" :value="m.id">
            {{ m.displayName }}{{ m.ready ? '' : ' — UNAVAILABLE' }}
          </option>
        </select>
      </div>
    </div>

    <p v-if="message" role="alert" class="text-xs text-red-400 bg-red-950/40 border border-red-900 rounded-md px-3 py-2 font-sans">
      {{ message }}
    </p>
  </div>
</template>
