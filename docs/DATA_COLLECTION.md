# Data Collection Protocol

Status: NOT STARTED — no recordings collected.

## Locked scope rule (2026-08-21)

Mentra recordings collected in this sprint are **evaluation / calibration /
debugging data only** — never required training data. All model
training/fine-tuning must draw from public datasets approved for the
experiment (see `docs/ARCHITECTURE.md` training-data list: LibriSpeech,
DNS5 Personalized, CHiME-9 ECHI, EasyCom — CHiME/EasyCom are research-license
only, check `LICENSES.md` before any commercial use). This section still
governs how we collect and split the Mentra-domain eval/calibration set.

## Session-disjoint splitting rule

Never split adjacent chunks of one recording across train/calibration/test.
Split by session, by environment, and (for environmental speakers) by
speaker identity. Calibration set and final test set must be disjoint.

## Minimum coverage per sprint spec section 17

Wearer: quiet/normal/loud, short/long utterances, varied phrases.
Environment: front/beside/behind, near/far, multiple speakers/genders.
Acoustic: quiet room, office, HVAC, music, TV, cafe, outside, traffic, wind.
Hard cases: rapid alternation, close environmental speaker, soft wearer +
loud environment, similar-pitch speaker, TV background, speakerphone,
overlapping speech.

## Manifest format

```json
{
  "file": "session03_0017.wav",
  "label": "WEARER",
  "session": "03",
  "condition": "office_noise",
  "snr": null
}
```

One manifest per session in `evaluation/manifests/`.

## Enrollment capture

Must be recorded THROUGH the Mentra Live mic, not laptop/phone mic, unless a
controlled test proves cross-device enrollment holds. Test durations: 5s,
10s, 20s, 30s, multiple utterances. No other speaker in enrollment audio.
