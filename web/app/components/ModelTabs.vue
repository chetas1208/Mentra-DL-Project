<script setup lang="ts">
// Segmented model tabs — visibly obvious SpeakerNet | GeoWearNet G2 in the
// existing control strip. Same panel styling as AudioSourceSelector; catalog-
// driven readiness, experimental flag and unavailable state from backend only.
import type { CatalogModel } from '~/utils/modelCatalog'

const props = defineProps<{
  models: CatalogModel[]
  selectedModel: string | null
  activeModel: string | null
  /** True while a session is live — switching requires STOP first. */
  disabled: boolean
  error: string | null
}>()
const emit = defineEmits<{ select: [modelId: string] }>()

function onSelect(modelId: string, ready: boolean) {
  if (props.disabled || !ready || modelId === props.selectedModel) return
  emit('select', modelId)
}

function onKeydown(e: KeyboardEvent, modelId: string, ready: boolean) {
  if (e.key === 'Enter' || e.key === ' ') {
    e.preventDefault()
    onSelect(modelId, ready)
  }
}

const selected = computed(() =>
  props.models.find((m) => m.id === props.selectedModel) ?? null,
)

const showExperimental = computed(
  () => selected.value?.ready === true && selected.value?.experimental === true,
)

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
      <span id="runtime-model-label" class="text-[11px] text-ink-dim tracking-wide shrink-0">MODEL</span>
      <span
        v-if="showExperimental"
        class="text-[10px] font-mono text-amber-400 shrink-0"
        role="status"
      >
        Experimental
      </span>
    </div>

    <div
      role="tablist"
      aria-labelledby="runtime-model-label"
      class="flex w-full rounded-md border border-panel-line overflow-hidden divide-x divide-panel-line"
    >
      <button
        v-for="m in models"
        :key="m.id"
        type="button"
        role="tab"
        :id="`model-tab-${m.id}`"
        :aria-selected="selectedModel === m.id"
        :aria-disabled="disabled || !m.ready"
        :tabindex="selectedModel === m.id ? 0 : -1"
        :disabled="disabled || !m.ready"
        class="flex-1 min-h-[44px] min-w-0 px-1.5 sm:px-2 py-2.5 text-[10px] sm:text-xs font-semibold tracking-wide transition-colors focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-wearer focus-visible:ring-inset disabled:cursor-not-allowed"
        :class="[
          selectedModel === m.id
            ? 'bg-panel-bg text-ink'
            : 'bg-panel-surface text-ink-dim hover:text-ink hover:bg-panel-bg/50',
          !m.ready ? 'opacity-40' : '',
          disabled ? 'opacity-50' : '',
        ]"
        @click="onSelect(m.id, m.ready)"
        @keydown="onKeydown($event, m.id, m.ready)"
      >
        <span class="block truncate leading-tight">{{ m.displayName }}</span>
        <span v-if="!m.ready" class="block text-[9px] font-normal text-red-400/80 mt-0.5 normal-case">
          unavailable
        </span>
      </button>
      <div
        v-if="models.length === 0"
        class="flex-1 min-h-[44px] px-2 py-2.5 text-xs text-ink-faint flex items-center justify-center"
        role="status"
      >
        NOT AVAILABLE
      </div>
    </div>

    <p
      v-if="message"
      role="alert"
      class="text-xs text-red-400 bg-red-950/40 border border-red-900 rounded-md px-3 py-2 font-sans"
    >
      {{ message }}
    </p>
  </div>
</template>
