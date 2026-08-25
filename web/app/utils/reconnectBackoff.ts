// Bounded exponential backoff with jitter (spec section 21, RECONNECT).
// Pure function so it's deterministically testable -- no timers, no
// randomness hidden inside a class with side effects.

export interface BackoffConfig {
  baseMs: number
  maxMs: number
  maxAttempts: number
}

export const DEFAULT_BACKOFF: BackoffConfig = { baseMs: 500, maxMs: 15_000, maxAttempts: 8 }

/** attempt is 1-based (first retry = attempt 1). Returns null once
 * maxAttempts is exceeded -- caller must stop retrying, not loop forever. */
export function nextBackoffDelayMs(attempt: number, config: BackoffConfig, random: () => number = Math.random): number | null {
  if (attempt < 1 || attempt > config.maxAttempts) return null
  const exp = Math.min(config.maxMs, config.baseMs * 2 ** (attempt - 1))
  // full jitter: uniform in [0, exp] -- avoids synchronized reconnect storms
  // across multiple tabs/clients better than a fixed +/- jitter window.
  return Math.round(random() * exp)
}
