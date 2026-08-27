"""G4 WS7-38 -- the complete, structured BLOCKED_NO_MENTRA_HARDWARE registry.

Each entry states three things and nothing vaguer:
  * why it is blocked (what physical thing is missing),
  * the EXACT unblock requirement (device, people, minutes, conditions),
  * which script already built in this repo will consume the data the moment
    it lands, so that no further engineering stands between a capture session
    and a result.

This is a data module on purpose: the report renders from it and
``tests/geowearnet/test_g4_blocked_registry.py`` checks it stays complete and
actionable, so an entry cannot rot into "TODO: needs hardware".
"""
from __future__ import annotations

from typing import Dict, List

BLOCK_REASON = "BLOCKED_NO_MENTRA_HARDWARE"

# The one physical prerequisite almost everything below shares.
BASE_HARDWARE_REQUIREMENT = {
    "device": "1x physical Mentra Live (or another MentraOS-compatible glasses model, recorded as such)",
    "phone": "1x paired phone running the MentraOS app; model and OS version recorded",
    "operator_machine": (
        "a laptop with a browser (Chrome/Edge) that can select the glasses microphone as an "
        "input device, OR a phone-side app built on the Mentra Bluetooth SDK if the browser "
        "route cannot reach the glasses mic"
    ),
    "software_already_ready": [
        "web/app/pages/capture.vue (research capture mode)",
        "scripts/mentra/run_receiver.py --capture-dir <dir>",
        "mentra/capture/ (session writer + automatic validation)",
    ],
    "account_or_credentials": (
        "none needed for the browser capture route. A MentraOS developer account and the "
        "Bluetooth SDK are only needed if the browser cannot reach the glasses microphone."
    ),
    "must_record_per_session": [
        "declared source_kind = MENTRA_LIVE_GLASSES_MIC",
        "glasses model, glasses firmware version, MentraOS app version",
        "granted AGC / noise-suppression / echo-cancellation flags (the capture page reads these back)",
        "native sample rate as reported by the capture chain -- never assumed",
    ],
}

# The first pilot everything else depends on. Small-N-first, consistent with
# how G2/G3 sequenced their own evidence.
PILOT_SMOKE = {
    "people": 5,
    "minutes_each": "5-10",
    "total_minutes": "25-50",
    "constraints": [
        "no voice enrollment of any kind",
        "camera OFF",
        "IMU OFF / unused",
        "the SAME physical glasses passed between all five people (this is the shared-glasses test)",
        "each person records at least one quiet-solo take and one take with a bystander talking",
    ],
    "conditions_per_person": [
        "quiet_solo -- wearer speaks alone in a quiet room",
        "bystander_1m -- a second person talks ~1 m away, including deliberate overlap",
        "noise_machinery -- continuous broadband/impulsive noise (a shop vac, fan, or actual tools)",
        "quiet_wearer -- wearer speaks softly / turns their head away mid-utterance",
        "agent_commands -- 10 scripted commands per person, spoken by the wearer, plus 10 of the "
        "SAME commands spoken by the bystander (this is what makes false-command-rate measurable)",
    ],
}


def entries() -> List[Dict[str, object]]:
    """The brief's WS7-38 range enumerates the 21 named workstreams below.

    All require physical hardware, a paired phone, and/or human pilot
    subjects. None was attempted with substitute data.
    """
    return [
        {
            "id": "G4-B01",
            "workstream": "Five-person zero-shot smoke test",
            "status": BLOCK_REASON,
            "why": "Needs five different people physically wearing the device. No device, no people.",
            "needs": {"pilot": PILOT_SMOKE},
            "consumes": "captures/<session>/ written by web/app/pages/capture.vue + run_receiver.py --capture-dir",
            "then_run": "python3 -m training.geowearnet.g4.parity (point it at capture sessions instead of MMCSG items)",
            "engineering_remaining": "small: an EvalItem adapter from a capture session directory",
        },
        {
            "id": "G4-B02",
            "workstream": "Shared-glasses test (device passed between wearers)",
            "status": BLOCK_REASON,
            "why": "The software half is done and tested; the physical half needs one device and several heads.",
            "needs": {"pilot": PILOT_SMOKE,
                      "protocol": "one continuous receiver process; each wearer is a separate transport session"},
            "consumes": "one capture session per wearer, same glasses, different anonymous wearer ids",
            "then_run": "compare per-wearer detector behaviour across sessions; no code change needed to swap wearers",
            "engineering_remaining": "none for the software reset path -- already covered by "
                                     "tests/audio/test_frontend_envelope.py and tests/audio/test_session_reset.py",
        },
        {
            "id": "G4-B03",
            "workstream": "G4 zero-shot evaluation of the frozen G2 parent on Mentra audio",
            "status": BLOCK_REASON,
            "why": "The headline G4 question. Requires real Mentra microphone audio, which does not exist here.",
            "needs": {"pilot": PILOT_SMOKE,
                      "labels": "per-frame wearer/bystander activity. Cheapest honest source: record each "
                                "speaker on a separate close mic simultaneously, or hand-annotate 25-50 minutes "
                                "(~4-8 hours of annotation)"},
            "consumes": "capture sessions + activity labels",
            "then_run": "evaluation/agent_audio/run_matrix.py with a Mentra item source",
            "engineering_remaining": "small: a labelled-item adapter; the matrix, metrics and gate are unchanged",
        },
        {
            "id": "G4-B04",
            "workstream": "Pipeline matrix (RAW / RNNOISE / GATE / ORACLE) on real Mentra audio",
            "status": BLOCK_REASON,
            "why": "Same data dependency as G4-B03; oracle rows additionally need ground-truth activity.",
            "needs": {"pilot": PILOT_SMOKE, "labels": "same as G4-B03"},
            "consumes": "labelled Mentra capture sessions",
            "then_run": "evaluation/agent_audio/run_matrix.py",
            "engineering_remaining": "none beyond the G4-B03 adapter",
        },
        {
            "id": "G4-B05",
            "workstream": "Agent command bench on real hardware",
            "status": BLOCK_REASON,
            "why": "Needs people speaking scripted commands into the real device.",
            "needs": {"pilot": PILOT_SMOKE,
                      "extra": "the scripted command list must be spoken by BOTH the wearer and the bystander"},
            "consumes": "capture sessions with the agent_commands condition",
            "then_run": "evaluation/agent_audio/commands.py via run_matrix.py",
            "engineering_remaining": "none",
        },
        {
            "id": "G4-B06",
            "workstream": "Autobody-shop audio conditions on real hardware",
            "status": BLOCK_REASON,
            "why": "The synthetic stress bench is a proxy. Real impact wrenches, compressors and shop "
                   "reverberation cannot be substituted with mixed corpus noise and called a hardware result.",
            "needs": {"pilot": PILOT_SMOKE,
                      "location": "an actual workshop or a space with comparable machinery noise and reverberation",
                      "minutes": "10-15 additional minutes per wearer in the noisy space"},
            "consumes": "capture sessions labelled with the noise condition",
            "then_run": "evaluation/agent_audio/run_matrix.py, compared against the synthetic stress bench",
            "engineering_remaining": "none",
        },
        {
            "id": "G4-B07",
            "workstream": "Distance and quiet-wearer stress with real microphones",
            "status": BLOCK_REASON,
            "why": "Bystander distance and wearer vocal effort are physical variables. Simulating them "
                   "would be measuring the simulator's room model, which G1 already did.",
            "needs": {"pilot": PILOT_SMOKE,
                      "protocol": "bystander at 0.5 m / 1 m / 3 m; wearer at normal / soft / head-turned-away"},
            "consumes": "capture sessions with distance and effort recorded in the condition label",
            "then_run": "evaluation/agent_audio/run_matrix.py grouped by condition",
            "engineering_remaining": "none",
        },
        {
            "id": "G4-B08",
            "workstream": "Real agent ASR on real Mentra captures",
            "status": BLOCK_REASON,
            "why": "WER on Mentra audio requires Mentra audio and reference transcripts.",
            "needs": {"pilot": PILOT_SMOKE,
                      "labels": "word-level reference transcripts per speaker (scripted prompts make this cheap)"},
            "consumes": "capture sessions + transcripts",
            "then_run": "evaluation/agent_audio/asr.py + metrics.py (the same streaming ASR backend used in G3)",
            "engineering_remaining": "none",
        },
        {
            "id": "G4-B09",
            "workstream": "False-agent-command rate on real captures",
            "status": BLOCK_REASON,
            "why": "Requires a bystander physically speaking commands the agent must NOT act on.",
            "needs": {"pilot": PILOT_SMOKE, "extra": "same command list spoken by the bystander"},
            "consumes": "capture sessions with bystander_commands recorded",
            "then_run": "evaluation/agent_audio/commands.py",
            "engineering_remaining": "none",
        },
        {
            "id": "G4-B10",
            "workstream": "Mentra zero-shot verdict",
            "status": BLOCK_REASON,
            "why": "A verdict is a conclusion from G4-B03/B04/B05; with no inputs there is no verdict, "
                   "and inventing one would be the exact overclaim this project forbids.",
            "needs": {"depends_on": ["G4-B03", "G4-B04", "G4-B05"]},
            "consumes": "the results of those three",
            "then_run": "training/geowearnet/g4/finalize.py",
            "engineering_remaining": "none",
        },
        {
            "id": "G4-B11",
            "workstream": "Domain-gap autopsy against real Mentra distributions",
            "status": BLOCK_REASON,
            "why": "A gap needs two distributions. The MMCSG side exists; the Mentra side does not.",
            "needs": {"pilot": PILOT_SMOKE,
                      "note": "the raw native-rate stream matters most here -- a resampled-only capture "
                              "would measure the resampler, not the device"},
            "consumes": "captures/<session>/raw.wav at the true native rate",
            "then_run": "a feature-distribution comparison against the MMCSG stats already cached in "
                        "training/geowearnet/mmcsg/cache/",
            "engineering_remaining": "moderate: a distribution-comparison script (not yet written, "
                                     "deliberately, because its design depends on what the real format turns out to be)",
        },
        {
            "id": "G4-B12",
            "workstream": "Small Mentra domain adaptation",
            "status": "NOT_APPLICABLE_" + BLOCK_REASON,
            "why": "There is nothing to adapt to. Fine-tuning on substitute audio and calling the result "
                   "Mentra-adapted would be fabrication.",
            "needs": {"pilot": "the smoke pilot plus enough labelled minutes to justify training -- "
                               "G2 measured the real-data sample-efficiency curve plateauing near 60 minutes, "
                               "so budget 60+ labelled minutes before training anything"},
            "consumes": "labelled Mentra capture sessions",
            "then_run": "training/geowearnet/mmcsg/train_real.py initialised from the frozen G2 parent",
            "engineering_remaining": "small: a Mentra dataset adapter mirroring the MMCSG one",
            "gate": "do not start until the zero-shot verdict (G4-B10) shows adaptation is actually needed",
        },
        {
            "id": "G4-B13",
            "workstream": "Mentra learning curve (minutes-of-data vs performance)",
            "status": BLOCK_REASON,
            "why": "A learning curve needs several dataset sizes of real Mentra data.",
            "needs": {"pilot": "60+ labelled minutes, sliceable into 5/10/20/40/60-minute budgets"},
            "consumes": "labelled Mentra capture sessions",
            "then_run": "the same budget-sweep harness G2 used for its MMCSG sample-efficiency curve",
            "engineering_remaining": "small, follows G4-B12",
        },
        {
            "id": "G4-B14",
            "workstream": "Cross-wearer generalisation on Mentra",
            "status": BLOCK_REASON,
            "why": "Needs multiple real wearers, held out from each other.",
            "needs": {"pilot": PILOT_SMOKE,
                      "note": "5 people is enough to detect a large effect, not to estimate a small one; "
                              "treat the first result as a smoke test, not a generalisation claim"},
            "consumes": "capture sessions grouped by anonymous wearer id",
            "then_run": "a wearer-disjoint split evaluation, mirroring the G2/G3 internal-val protocol",
            "engineering_remaining": "small",
        },
        {
            "id": "G4-B15",
            "workstream": "Hardware session variability (same wearer, different sessions)",
            "status": BLOCK_REASON,
            "why": "Requires re-donning the same physical glasses on different occasions.",
            "needs": {"pilot": "at least 2 people recording the SAME conditions on 2 separate days, "
                               "re-seating the glasses each time"},
            "consumes": "capture sessions sharing a wearer id across dates",
            "then_run": "within-wearer between-session comparison",
            "engineering_remaining": "small",
        },
        {
            "id": "G4-B16",
            "workstream": "Multi-microphone question",
            "status": "PARTIALLY_ANSWERED_BY_DESK_RESEARCH",
            "why": "Public docs answer the API half; only the device answers the rest.",
            "desk_research_finding": (
                "Mentra Live is publicly described as having multiple microphones, with one dedicated to an "
                "LC3-compressed app channel. No public API exposes multi-channel or per-microphone audio: "
                "the SDK offers a single mic stream plus setPreferredMic() to choose a source. "
                "See docs/geowearnet_g4_mentraos_audio_path.md."
            ),
            "needs": {"device": "the device plus the Bluetooth SDK, to confirm no multi-channel path exists"},
            "consumes": "device inspection",
            "then_run": "n/a -- this is an inspection, not an evaluation",
            "engineering_remaining": "none",
            "implication": "G2's +0.05-0.06 AUROC from MMCSG multichannel spatial features must NOT be "
                           "assumed to transfer to Mentra Live.",
        },
        {
            "id": "G4-B17",
            "workstream": "Overlap re-evaluation on real Mentra audio",
            "status": BLOCK_REASON,
            "why": "G3 measured overlap contributing 6.90% of oracle residual leakage on MMCSG. Whether "
                   "that holds on Mentra's own microphone and geometry is unknown and unmeasurable here.",
            "needs": {"pilot": PILOT_SMOKE, "labels": "overlap-marked activity labels"},
            "consumes": "labelled Mentra capture sessions",
            "then_run": "evaluation/agent_audio/overlap.py",
            "engineering_remaining": "none beyond the G4-B03 adapter",
        },
        {
            "id": "G4-B18",
            "workstream": "WearerSepNet separator-gate re-check with real evidence",
            "status": BLOCK_REASON,
            "why": "The gate is CLOSED on G3's evidence. Reopening it requires new evidence, and the only "
                   "evidence that would count is real Mentra overlap measurement.",
            "needs": {"depends_on": ["G4-B17"]},
            "consumes": "the overlap re-evaluation result",
            "then_run": "the same predeclared go/no-go: overlap must cause >=50% of residual leakage AND "
                        "muting it must cost <5pp wearer deletion",
            "engineering_remaining": "none",
            "current_state": "CLOSED -- unchanged from G3. No new evidence was produced in G4.",
        },
        {
            "id": "G4-B19",
            "workstream": "Live receiver integration test against a real device",
            "status": BLOCK_REASON,
            "why": "The receiver has been exercised end-to-end against the real transport with synthetic "
                   "audio (scripts/mentra/capture_selftest.py). What is untested is a real device as the "
                   "audio source: real clock drift, real dropouts, real Bluetooth jitter.",
            "needs": {"pilot": "one 10-minute continuous session with the device connected through the "
                               "deployed web app or a local dev instance"},
            "consumes": "a live session against run_receiver.py --capture-dir",
            "then_run": "inspect jitter-buffer stats, discontinuity flags and dropped-frame counters the "
                        "receiver already records",
            "engineering_remaining": "none -- the telemetry already exists",
        },
        {
            "id": "G4-B20",
            "workstream": "Real end-to-end latency (capture -> transport -> inference -> agent)",
            "status": BLOCK_REASON,
            "why": "Compute latency is measured (see the CPU artifacts). Capture and transport latency are "
                   "properties of the physical device, the Bluetooth link and the operator's network.",
            "needs": {"pilot": "one session with the device, ideally through the deployed tunnel so the "
                               "real network hop is included"},
            "consumes": "the round-trip timestamps the transport already echoes back to the browser",
            "then_run": "the existing captureToPredictionMs telemetry in web/app/composables/useMentraTransport.ts",
            "engineering_remaining": "none",
            "already_known": "software compute is real-time with large margin; see "
                             "evaluation/geowearnet/g4/routing_cpu_benchmark.json and the cadence sweep "
                             "in evaluation/geowearnet/g4/live_offline_parity.json",
        },
        {
            "id": "G4-B21",
            "workstream": "Glasses-swap soak test on real hardware",
            "status": BLOCK_REASON,
            "why": "A soak on real hardware means hours of a real device connected and swapped between "
                   "real people. A software soak proves no leak or drift in this process, not that the "
                   "device stays connected.",
            "needs": {"pilot": "a 60+ minute continuous receiver run with at least 6 wearer swaps"},
            "consumes": "one long-running receiver process with per-session capture directories",
            "then_run": "watch RSS, NaN/Inf counts, and per-session reset logs, as G3's soak already does",
            "engineering_remaining": "none",
            "software_equivalent_already_done": (
                "G3 streamed 6.15 real minutes with 0 NaN/Inf and 0.78 MB RSS growth; G4 additionally "
                "asserts envelope/router state is fully reset between sessions "
                "(tests/audio/test_frontend_envelope.py). Neither proves anything about the physical device."
            ),
        },
    ]


def summary() -> Dict[str, object]:
    rows = entries()
    return {
        "n_blocked_workstreams": len(rows),
        "block_reason": BLOCK_REASON,
        "note": ("The brief's WS7-38 range enumerates these 21 named workstreams. None was attempted "
                 "with substitute, synthetic, or corpus audio presented as Mentra data."),
        "base_hardware_requirement": BASE_HARDWARE_REQUIREMENT,
        "first_pilot": PILOT_SMOKE,
        "entries": rows,
        "single_highest_value_unblock": (
            "One physical Mentra Live + a paired phone + five people for 5-10 minutes each, captured "
            "through web/app/pages/capture.vue against run_receiver.py --capture-dir. That single "
            "session unblocks G4-B01, B02, B03, B19, B20 immediately, and B04-B09/B17 as soon as the "
            "takes are labelled."
        ),
    }
