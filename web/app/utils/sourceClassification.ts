// Source classification (spec section 6) -- derives a DISPLAY label from
// real MediaDeviceInfo metadata. This is a LABEL, not hardware
// authentication: it does not grant/verify any capability, it just tells
// an operator what the browser told us about the selected input.

export type SourceClassification = 'MENTRA LIVE' | 'BROWSER MICROPHONE' | 'UNKNOWN AUDIO INPUT'

/** `label` may be '' if microphone permission hasn't been granted yet --
 * per section 4, an unlabeled device must NEVER be classified as Mentra,
 * it must read UNKNOWN until a real label is available. */
export function classifySource(label: string | undefined | null): SourceClassification {
  if (!label) return 'UNKNOWN AUDIO INPUT'
  if (/mentra/i.test(label)) return 'MENTRA LIVE'
  return 'BROWSER MICROPHONE'
}
