/**
 * E2E: multiple audio INPUT devices (microphones) on the live console.
 * NOT SpeakerNet — tests the INPUT selector, device switching, and
 * input-lost handling with mocked enumerateDevices/getUserMedia.
 *
 * Run (dev server on 3001):
 *   cd web && npx tsx tests/e2e/multiAudioInput.e2e.ts
 */
import { firefox, type BrowserContext, type Page } from 'playwright'

const BASE = process.env.MENTRA_WEB_URL ?? 'http://127.0.0.1:3001'

const FAKE_INPUTS = [
  { deviceId: 'mic-macbook', kind: 'audioinput', label: 'MacBook Pro Microphone', groupId: 'g1' },
  { deviceId: 'mic-usb-yeti', kind: 'audioinput', label: 'Blue Yeti USB', groupId: 'g2' },
  { deviceId: 'mic-mentra', kind: 'audioinput', label: 'Mentra Live Microphone', groupId: 'g3' },
  { deviceId: 'spk-out', kind: 'audiooutput', label: 'Built-in Output', groupId: 'g4' },
]

async function injectFakeMediaDevices(page: Page, inputs = FAKE_INPUTS) {
  await page.addInitScript((devices) => {
    const audioInputs = devices.filter((d: { kind: string }) => d.kind === 'audioinput')

    class FakeMediaStreamTrack {
      kind = 'audio'
      label: string
      enabled = true
      readyState: MediaStreamTrackState = 'live'
      constructor(label: string) { this.label = label }
      stop() { this.readyState = 'ended' }
      getSettings() {
        return {
          deviceId: audioInputs[0]?.deviceId ?? 'default',
          sampleRate: 48000,
          channelCount: 1,
          echoCancellation: false,
          noiseSuppression: false,
          autoGainControl: false,
        }
      }
    }

    const orig = navigator.mediaDevices
    navigator.mediaDevices.enumerateDevices = async () => devices as MediaDeviceInfo[]
    navigator.mediaDevices.getUserMedia = async (constraints) => {
      const wanted = (constraints as MediaStreamConstraints)?.audio
      let deviceId = audioInputs[0]?.deviceId
      if (wanted && typeof wanted === 'object' && wanted.deviceId) {
        const req = wanted.deviceId
        deviceId = typeof req === 'string' ? req : req.exact
      }
      const label = audioInputs.find((d: { deviceId: string }) => d.deviceId === deviceId)?.label ?? 'Fake Mic'
      const track = new FakeMediaStreamTrack(label) as unknown as MediaStreamTrack
      return { getAudioTracks: () => [track], getTracks: () => [track] } as MediaStream
    }
    // Preserve addEventListener for devicechange wiring
    if (orig?.addEventListener) {
      navigator.mediaDevices.addEventListener = orig.addEventListener.bind(orig)
    }
  }, inputs)
}

async function grantMic(context: BrowserContext) {
  await context.grantPermissions(['microphone'], { origin: BASE })
}

async function getInputOptions(page: Page): Promise<string[]> {
  return page.locator('#input-device option').allTextContents()
}

async function selectInput(page: Page, deviceId: string) {
  await page.selectOption('#input-device', deviceId)
}

async function main() {
  const browser = await firefox.launch({ headless: true })
  const context = await browser.newContext()
  await grantMic(context)
  const page = await context.newPage()

  const results: { name: string; pass: boolean; detail?: string }[] = []

  function record(name: string, pass: boolean, detail?: string) {
    results.push({ name, pass, detail })
    console.log(`${pass ? '✓' : '✗'} ${name}${detail ? ` — ${detail}` : ''}`)
  }

  try {
    // --- Live dashboard: multiple inputs in dropdown --------------------
    await injectFakeMediaDevices(page)
    await page.goto(BASE, { waitUntil: 'networkidle' })

    // Trigger permission refresh path — page calls refreshDevices on mount
    // after connect attempt; force a re-enumeration via devicechange
    await page.evaluate(() => {
      navigator.mediaDevices.dispatchEvent(new Event('devicechange'))
    })
    await page.waitForTimeout(500)

    const options = await getInputOptions(page)
    record(
      'INPUT dropdown lists all audioinput devices',
      options.includes('MacBook Pro Microphone')
        && options.includes('Blue Yeti USB')
        && options.includes('Mentra Live Microphone')
        && !options.some((o) => o.includes('Built-in Output')),
      `options: ${options.join(' | ')}`,
    )

    // Switch to USB mic
    await selectInput(page, 'mic-usb-yeti')
    await page.waitForTimeout(200)
    const usbClass = await page.locator('label[for="input-device"]')
      .locator('..')
      .locator('.font-mono')
      .first()
      .textContent()
    record(
      'Selecting USB mic shows BROWSER MICROPHONE classification',
      usbClass?.includes('BROWSER MICROPHONE') ?? false,
      usbClass ?? '',
    )

    // Switch to Mentra mic
    await selectInput(page, 'mic-mentra')
    await page.waitForTimeout(200)
    const mentraClass = await page.locator('label[for="input-device"]')
      .locator('..')
      .locator('.font-mono')
      .first()
      .textContent()
    record(
      'Selecting Mentra mic shows MENTRA LIVE classification',
      mentraClass?.includes('MENTRA LIVE') ?? false,
      mentraClass ?? '',
    )

    // INPUT selector disabled while capturing — check attribute wiring exists
    record(
      'INPUT select exists and is enabled while idle',
      await page.locator('#input-device').isEnabled(),
    )

    // --- Input lost: selected device disappears -------------------------
    await selectInput(page, 'mic-usb-yeti')
    await injectFakeMediaDevices(page, FAKE_INPUTS.filter((d) => d.deviceId !== 'mic-usb-yeti'))
    await page.goto(BASE, { waitUntil: 'networkidle' })
    await page.evaluate(() => navigator.mediaDevices.dispatchEvent(new Event('devicechange')))
    await page.waitForTimeout(500)

    const lostAlert = await page.locator('[role="alert"]').filter({ hasText: 'INPUT LOST' }).count()
    record(
      'Unplugged selected mic triggers INPUT LOST alert',
      lostAlert > 0,
    )

    // --- Audio debug page: lists all inputs as buttons ------------------
    await injectFakeMediaDevices(page)
    await page.goto(`${BASE}/audio-debug`, { waitUntil: 'networkidle' })
    await page.click('button:has-text("REQUEST MICROPHONE PERMISSION")')
    await page.waitForTimeout(800)

    const debugButtons = await page.locator('button:has-text("Microphone"), button:has-text("Yeti"), button:has-text("Mentra")').count()
    record(
      'Audio-debug page lists multiple input devices',
      debugButtons >= 3,
      `found ${debugButtons} device buttons`,
    )

    await page.click('button:has-text("Blue Yeti USB")')
    await page.waitForTimeout(300)
    const settingsVisible = await page.locator('text=REPORTED TRACK SETTINGS').isVisible()
    record(
      'Clicking an input on audio-debug shows track settings',
      settingsVisible,
    )

    await page.click('button:has-text("Mentra Live Microphone")')
    const mentraBadge = await page.locator('text=MENTRA ✓').count()
    record(
      'Mentra input highlighted on audio-debug',
      mentraBadge > 0,
      `${mentraBadge} Mentra badge(s)`,
    )

  } catch (e) {
    record('E2E runner', false, e instanceof Error ? e.message : String(e))
  } finally {
    await browser.close()
  }

  const failed = results.filter((r) => !r.pass)
  console.log('\n---')
  console.log(`${results.length - failed.length}/${results.length} passed`)
  if (failed.length) {
    console.error('FAILED:', failed.map((f) => f.name).join(', '))
    process.exit(1)
  }
}

main()
