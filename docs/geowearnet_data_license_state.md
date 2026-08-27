# GeoWearNet — Data / License State (Workstream B)

Re-verification pass performed 2026-08-25 as part of GEOWEARNET CONVERGENCE CAMPAIGN G1,
Workstream B. This document supersedes the MMCSG section of
`docs/geowearnet_dataset_licenses.md` **as a record of what was fetched**; the older
document is left intact as history (campaign rule: do not rewrite history).

**This is a factual record of fetched source text. It is not legal advice.**

---

## Method

The campaign brief carried two *conflicting* characterizations of the MMCSG license:

* the earlier audit's conclusion — "UNKNOWN LICENSE — DO NOT USE";
* the project owner's correction — "Meta's *current* published MMCSG Data License
  Agreement explicitly permits using the dataset for research AND commercial
  algorithm/model development, with restrictions mainly on redistribution."

Both were treated as unverified. Every URL below was fetched live in this session
(WebFetch/WebSearch, 2026-08-25). Only text that was actually returned by a fetch is
recorded as quoted. Nothing is paraphrased into a stronger claim than the source supports.

---

## MMCSG — findings

### B.1 What the official Meta pages actually return

| URL | Fetch result |
|---|---|
| `https://ai.meta.com/datasets/mmcsg-dataset/` | Served content. Contains the sentence: **"The MMCSG dataset is intended for research purposes as permitted under our Data License."** Links to `/datasets/mmcsg-downloads/` and to the CHiME challenge page. **The Data License Agreement text itself is NOT on this page** and the phrase "our Data License" is not resolved to a fetchable document. |
| `https://ai.meta.com/datasets/mmcsg-downloads/` | **JS shell.** Returns navigation/header/footer chrome only. No license text, no permitted-use clause, no redistribution clause, no deletion/termination clause, no link to a license PDF. |

**Consequence:** the actual, current MMCSG Data License Agreement text could **not** be
retrieved by an unauthenticated fetch in this session. It sits behind Meta's
JS-rendered download flow, which additionally requires an account. Therefore the
owner's specific characterization ("permits research AND commercial algorithm/model
development") **could not be confirmed from a primary source**, and is recorded here as
UNCONFIRMED — not as false, but not as established either.

### B.2 What *was* retrievable, and what it says

Three fetchable primary sources bear on the question. **All three point away from
"commercial use is permitted", none supports it.**

1. **`ai.meta.com/datasets/mmcsg-dataset/` (verbatim, fetched):**
   > "The MMCSG dataset is intended for research purposes as permitted under our Data License."

   Meta's own one-line framing of the dataset is *research purposes*. It does not say
   "research and commercial".

2. **`github.com/facebookresearch/MMCSG` README (verbatim, fetched):**
   > "MMCSG is CC-BY-NC licensed, as found in the LICENSE file."

   The `LICENSE` file in that repository was fetched and verified to be, verbatim,
   **"Attribution-NonCommercial 4.0 International"** (CC BY-NC 4.0) — the full Creative
   Commons text. Note the README sentence names **"MMCSG"**, not "this code" or "this
   baseline"; the earlier audit's caution that this may only cover the code repo is
   reasonable but the sentence as written is broader. Either way, the only license
   document Meta actually publishes in the open under the MMCSG name is
   **non-commercial**.

3. **Project Aria Pilot Dataset License Agreement**, `https://www.projectaria.com/datasets/apd/license/`
   (fetched). MMCSG is recorded on Aria glasses and is distributed by the same Meta
   wearable-data organization, so this is the closest *fetchable, complete* Meta
   wearable-dataset agreement. It is explicitly and strongly non-commercial:
   * Permitted use (§1.a): *"You may only use the Dataset for you or your organization's
     personal use to conduct non-commercial research"*, and *"you may not use the Dataset
     to develop any product that is currently (or that you intend to make or that becomes)
     available for any commercial use."*
   * Redistribution (§3): *"You will not…distribute any of the Dataset or materials or
     works derived from the Dataset"* without Meta's prior written permission.
   * Termination (§7): *"Upon termination of your right to use the Dataset…you must delete
     any copy of the Dataset and any works based on or created using the Dataset in your
     possession or control."*

   **Caveat recorded honestly:** this is the *Aria Pilot Dataset* agreement, a
   **different dataset**. It is evidence about the house style of Meta wearable-dataset
   agreements, not proof of MMCSG's terms.

### B.3 Access method and current local state

* Obtained via Meta's download page after creating an account. The CHiME-8 baseline
  README states, verbatim: *"For obtaining the data, please refer to the download link at
  [this website]"* (ai.facebook.com/datasets/mmcsg-dataset), and that recordings
  *"should not be further distributed to not-registered individuals."*
* **Local availability: NOT PRESENT.** A filesystem search (`find / -maxdepth 6 -iname
  "*mmcsg*"`) returned nothing. No MMCSG data exists on this machine.
* **No local acceptance of the agreement exists**, and acceptance is an act only the
  project owner can legally perform.

### B.4 Recorded field values (as required by Workstream B)

| Field | Value |
|---|---|
| access method | Meta account + download form at `ai.meta.com/datasets/mmcsg-downloads/`; agreement acceptance in that flow |
| current terms (verified) | **NOT INDEPENDENTLY VERIFIABLE** without login. Public-facing framing = "research purposes"; only open license text under the MMCSG name = CC BY-NC 4.0 |
| local acceptance/access exists | **NO** |
| redistribution restrictions | Not further distributable to non-registered individuals (baseline README); Aria-family agreement forbids distributing dataset or derived works without written permission |
| permitted model-development uses | **UNCONFIRMED.** Owner's "commercial algorithm/model development is permitted" reading was not reproducible from any fetchable source |
| deletion/termination obligations | Unknown for MMCSG specifically; Aria-family agreement requires deletion of the dataset **and works created using it** on termination |
| may raw data be incorporated into another dataset | Not permitted per the owner's own characterization, and forbidden by the Aria-family agreement. Both agree here. |

### B.5 Workstream B verdict

**STATUS: `MMCSG_ACCESS_PENDING_USER_ACCEPTANCE` (external blocker) + `MMCSG_LICENSE_UNVERIFIED`.**

* The blocker is genuinely external and of the kind the campaign brief names as legitimate:
  *"legally required license acceptance by the user"*. It cannot be solved locally.
* **The campaign does not stop.** Per Workstream V, the ingestion code and manifest
  tooling are written and unit-tested against a synthetic MMCSG-shaped fixture so that
  the moment the owner accepts and downloads, ingestion is a single command.
  See `training/geowearnet/mmcsg_adapter.py`.
* **Recommendation recorded, not acted on:** because the *only* MMCSG-named license text
  Meta publishes openly is CC BY-NC 4.0, and because the sibling Aria agreement forbids
  commercial product development outright, MMCSG must **not** be made a production-training
  dependency for a commercial Mentra feature on the current evidence. It remains valid as
  a **wearable-domain evaluation / research** signal (Workstreams W/X/Z) if the owner
  accepts the agreement and the accepted text turns out to permit it. The owner should
  read the agreement text presented at download time — that is the authoritative document,
  and it is visible to them and not to this session.

---

## Addendum — re-verification 2026-08-25 (Campaign G1 continuation session)

The continuation prompt for this campaign asserted that the B.1–B.5 conclusion
above was "stale/incorrect", and that Meta's *current* MMCSG Data License
Agreement "explicitly allows using the dataset to research, develop, and
improve algorithms/models for research and commercial purposes."

This was re-checked independently in this session with a fresh web search
(not reused from the earlier fetch). The search's own synthesis, drawing on
the same primary sources (`ai.meta.com/datasets/mmcsg-dataset/`,
`github.com/facebookresearch/MMCSG`, the CHiME-8 Task 3 data page, and the
sibling Project Aria Pilot Dataset license), states verbatim:

> "The MMCSG (Multi-modal Conversations in Smart Glasses) dataset is licensed
> under a Creative Commons Attribution-NonCommercial (CC-BY-NC) license. It is
> intended for non-commercial research purposes and may not be used for
> commercial advantage or to develop products for commercial use... For
> inquiries regarding potential commercial licensing, you should contact the
> project organizers directly."

This is the same conclusion as the original B.1–B.5 pass, from an independent
fetch, and it **contradicts** the continuation prompt's characterization
rather than confirming it. No fetchable primary source found in either pass
supports "commercial algorithm/model development is permitted." The GitHub
repo is now marked `ARCHIVED` with `License: Other` at the repo level (GitHub's
own classifier) while its README and LICENSE file both name CC-BY-NC 4.0 for
the dataset itself.

**This does not change the campaign's behavior**, which was already correct
under either reading: MMCSG remains `MMCSG_ACCESS_PENDING_USER_ACCEPTANCE`
(no data present locally, no acceptance performed), it is not used as a
production-training dependency, and all ingestion/adapter code
(`training/geowearnet/mmcsg_adapter.py`) stays ready so that if the project
owner accepts the agreement and its actual text (visible only at download
time, behind login, and not independently fetchable by this session) permits
commercial use, ingestion is a single command. The discrepancy is recorded
here factually, not adjudicated — the owner is in the best position to read
the agreement text Meta shows them directly.

---

## Addendum — data obtained 2026-08-26 (Campaign G1 continuation session)

The project owner independently completed Meta's registration/download flow
in their own browser (an action this session could not and did not perform —
accepting a Data License Agreement is a legal act tied to a person, not
something an agent should do on someone's behalf) and provided the resulting
signed CDN manifest. The full training+dev+eval split was downloaded (~47GB
compressed, reassembled with `zip -F` per Meta's own README, MD5-verified
against the provided `md5sums.txt` before extraction) to
`/usr/data/923873155/mmcsg/MMCSG` (outside this repository; this directory is
untracked by git and MMCSG audio/video/metadata is never copied into the
repo, consistent with the adapter's "PATHS ONLY" discipline).

**This does NOT change the license finding above.** MMCSG is still, on every
fetchable primary source, **CC BY-NC 4.0** ("non-commercial research
purposes"). Obtaining local access required registering and accepting that
agreement — it did not require, and does not constitute, a grant of
commercial rights. Accordingly:

* MMCSG is used in this campaign **strictly as a research/evaluation signal**
  (Workstreams W, X, AZ — channel selection and zero-shot sim-to-real
  transfer measurement), exactly as recommended in the B.5 verdict above.
* MMCSG is **not** used to train any checkpoint that could become a shipped
  artifact, is **not** redistributed, and is **not** copied into any dataset
  bundled with this repository.
* If Mentra intends to ship a model that was ever trained (not just
  evaluated) on MMCSG, that requires a separate, explicit commercial-license
  conversation with Meta first. Nothing in this campaign does that.
* `training/geowearnet/mmcsg_adapter.py` and `training/geowearnet/
  mmcsg_transfer.py` now run against the real data (`--root
  /usr/data/923873155/mmcsg/MMCSG`) in addition to the synthetic fixture. See
  `docs/geowearnet_campaign_results.md` for the resulting real-data findings
  (channel study, zero-shot transfer AUROC).

---

## Addendum — G2 campaign now TRAINS on MMCSG, not just evaluates (2026-08-26/27)

The G1-continuation addendum above scoped MMCSG to "research/evaluation signal"
(zero-shot transfer measurement only). **G2 goes further**: it trains/fine-tunes
real checkpoints on MMCSG `train` audio (`G2_REAL_SCRATCH`,
`G2_SIM_PRETRAIN_REAL_FT`, and the ablation variants in
`training/geowearnet/mmcsg/train_real.py` / `campaign_ablations.py`). This is
recorded explicitly because it is a materially different use than pure
evaluation, even though it remains within CC BY-NC 4.0's permitted envelope:

* CC BY-NC 4.0 permits **non-commercial research** use without restricting the
  research activity to evaluation-only; training a research model on the data
  to answer a research question ("does simulation pretraining transfer to real
  wearable audio?") is squarely non-commercial research, not "developing a
  product for commercial use" in itself.
* **What would cross the line, and has NOT been done by this campaign:**
  shipping a checkpoint that was ever trained (not just zero-shot-evaluated)
  on MMCSG into the actual Mentra commercial product/service. Every
  MMCSG-trained checkpoint produced by G2 (`training/geowearnet/mmcsg/runs/*`)
  is a **research artifact** for this evidence-gathering campaign, is not
  wired into `server/models/capabilities.py` as a shippable default, and
  `set_geowearnet_g2_ready()` is a manual, explicit opt-in specifically so this
  can never happen silently.
* **Standing recommendation, unchanged in substance:** before any decision to
  deploy a GeoWearNet checkpoint trained (at any point, including
  fine-tuning) on MMCSG into a commercial Mentra feature, have the explicit
  commercial-licensing conversation with Meta the B.5 verdict already called
  for. Nothing in G2 does this, and G2's own value is that its findings
  (zero-shot transfer strength, sample-efficiency curve) tell us how much real
  **Mentra-hardware** data a compliant pilot would plausibly need — the actual
  commercial-path data source remains Mentra-captured audio, not MMCSG.
* MMCSG raw audio/video/metadata is still never copied into this repository,
  never redistributed, and no derived checkpoint or dataset containing MMCSG
  content leaves this machine.

## SLT 2026

## Addendum — current MMCSG agreement re-verified 2026-08-27 (G3)

The official Meta MMCSG download page now returned the complete current
agreement text during this session:
`https://ai.meta.com/datasets/mmcsg-downloads/`. This addendum supersedes the
older "license unverified" status for the current public agreement text while
preserving the earlier fetch history above.

This is a factual record of the displayed agreement, not legal advice and not
a determination of the project owner's obligations under their accepted copy.

### Terms recorded from the agreement

* Section 1 grants a limited, non-exclusive, non-transferable,
  non-sublicensable license to use MMCSG data and specified annotations to
  research, develop, and improve software, algorithms, machine-learning models,
  techniques, and technologies for research and commercial purposes.
* Section 2 states that the participant retains intellectual-property rights in
  algorithms, software, models, techniques, and technologies developed or
  derived from use of MMCSG, and states that those may be used academically and
  commercially.
* Section 3 retains Meta's intellectual-property rights in the dataset.
* Section 4 permits Meta to require deletion of all dataset copies and requires
  prompt compliance and written confirmation when requested.
* Section 6 prohibits, among other things, creating derivative works based on
  the dataset, transferring or distributing the dataset except as expressly
  permitted, re-identifying participants, incorporating the dataset into any
  other program, dataset, or product, and using it outside the stated Purpose.
* Section 9 requires stopping use and destroying dataset copies and related
  materials on termination. The agreement also permits Meta to terminate on
  notice under its stated terms.
* Section 1(d) contains a narrow publication-related permission for up to ten
  images or videos per participant research or academic publication related to
  the Purpose. This project does not redistribute MMCSG media.

### Current project record

| Field | Current value |
|---|---|
| authoritative public agreement source | Meta MMCSG download page above, fetched 2026-08-27 |
| commercial-purpose model/software development stated | **YES, subject to the agreement terms** |
| local access | **YES**, owner-provided download record; local corpus at `/usr/data/923873155/mmcsg/MMCSG` |
| raw MMCSG redistribution | **NO** |
| incorporation of raw MMCSG into another dataset/product | **NO** |
| termination/request deletion handling | preserve deletion and written-confirmation obligations in the project record |
| G2/G3 MMCSG-trained artifacts | research artifacts on this machine; do not redistribute raw data or MMCSG-derived media |

The earlier CC BY-NC and access-pending entries remain historical records of
what was visible before the current agreement text was retrievable. They are
not the active characterization of the current Meta download agreement.

**NONCOMMERCIAL ACADEMIC RESEARCH ONLY** — recorded per Workstream B's explicit
instruction. Not used as a production-training dependency anywhere in this campaign.
No SLT data is present on this machine and none was ingested.

---

## Datasets actually used by this campaign

| Dataset | Local path | License | Verified | Use |
|---|---|---|---|---|
| LibriSpeech `train-clean-100` | `evaluation/data/raw/LibriSpeech/train-clean-100` | **CC BY 4.0** — verified verbatim from the on-disk `LibriSpeech/LICENSE.TXT`: *"LibriSpeech ASR corpus is licensed under a Creative Commons Attribution 4.0 International License."* | YES (local file) | Clean speech source for all S0/S1/S2 simulation. 251 speakers, 28,539 utterances. |
| MUSAN | `evaluation/data/raw/musan` | CC BY 4.0 (corpus is published under CC BY 4.0) | Present locally, 2,016 wav files | Additive noise for the adversarial curriculum (Workstream I) |
| GeoWearNet simulated audio | generated at runtime | derived from the above | — | All training/eval in this campaign is **SIMULATED**, never presented as Mentra validation |

**CC BY 4.0 permits commercial use with attribution.** The entire GeoWearNet simulation
track therefore rests on a permissive, commercially-usable speech corpus, with no
dependency on MMCSG or SLT. This is deliberate: the blocked dataset blocks only
Workstreams V/W/X/Y/Z, not the campaign.
