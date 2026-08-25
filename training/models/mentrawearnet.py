"""MentraWearNet-v1 (sprint spec sections 2, 6-25, "GO" phase).

Architecture:

    ENROLLMENT (once, cached at deployment)
    reference audio -> SpeakerNetBackbone.encode_speaker() -> e_w [B,256]

    LIVE (every rolling window)
    waveform -> SpeakerNetBackbone.encode_frames() -> [B,1500,T]
             -> 1x1 projection -> [B,D,T]
             -> FiLM(e_w) + explicit similarity branch
             -> causal depthwise-separable TCN
             -> wearer_head / environment_head (independent sigmoid logits)

Two independent binary outputs, not a 4-way softmax, because wearer and
environment activity are not mutually exclusive (section 15/19 -- SILENCE/
WEARER/ENVIRONMENT/OVERLAP is derived from the pair (P_wearer, P_environment)
outside this module, not predicted as one of 4 classes).

Hard parameter budget: <10,000,000 for the DEPLOYMENT graph (backbone minus
the 7205-way training classifier, plus everything below). See
count_params.py for the enforced assertion.
"""
from __future__ import annotations

import torch
import torch.nn as nn

from training.models.speakernet_backbone import SpeakerNetBackbone


class FiLM(nn.Module):
    """gamma/beta close to identity at init (section 10) so the model starts
    close to the unconditioned backbone rather than destroying pretrained
    features on step 0."""

    def __init__(self, embedding_dim: int, feature_dim: int):
        super().__init__()
        self.to_gamma = nn.Linear(embedding_dim, feature_dim)
        self.to_beta = nn.Linear(embedding_dim, feature_dim)
        nn.init.zeros_(self.to_gamma.weight)
        nn.init.zeros_(self.to_gamma.bias)
        nn.init.zeros_(self.to_beta.weight)
        nn.init.zeros_(self.to_beta.bias)

    def forward(self, h: torch.Tensor, e_w: torch.Tensor) -> torch.Tensor:
        # h: [B, D, T]. e_w is EITHER:
        #   - [B, E]: static per-utterance embedding (original behavior,
        #     identical computation to before this branch existed) -- gamma/
        #     beta computed once, broadcast over T via unsqueeze(-1).
        #   - [B, E, T]: per-frame embedding (V3C-MA0 addition, only ever
        #     produced by MentraWearNet.process_with_embedding when an
        #     adapter is active -- see MixtureAwareEnrollmentAdapter). The
        #     SAME to_gamma/to_beta Linear layers are applied per-frame
        #     (Linear operates on the last dim, so we transpose to [B,T,E],
        #     apply, transpose back) -- no new parameters, no change to the
        #     2D path, so every existing call site (e_w always [B,E]) is
        #     byte-identical to before this change.
        if e_w.dim() == 2:
            gamma = self.to_gamma(e_w).unsqueeze(-1)  # [B, D, 1]
            beta = self.to_beta(e_w).unsqueeze(-1)
        else:
            gamma = self.to_gamma(e_w.transpose(1, 2)).transpose(1, 2)  # [B, D, T]
            beta = self.to_beta(e_w.transpose(1, 2)).transpose(1, 2)
        return (1.0 + gamma) * h + beta


class CausalDepthwiseSeparableBlock(nn.Module):
    """One TCN block: causal (left-pad only, never sees future samples).
    depthwise Conv1d -> pointwise Conv1d -> GroupNorm -> SiLU -> dropout ->
    residual. GroupNorm normalizes over channels only (not time), so it
    doesn't leak future-frame statistics into current predictions -- unlike
    e.g. a naive LayerNorm-over-time or BatchNorm computed across the batch's
    full temporal extent during eval mode with running stats disabled."""

    def __init__(self, channels: int, kernel_size: int, dilation: int, dropout: float = 0.1):
        super().__init__()
        self.left_pad = (kernel_size - 1) * dilation  # causal: pad only on the left
        self.depthwise = nn.Conv1d(channels, channels, kernel_size,
                                    dilation=dilation, groups=channels, bias=False)
        self.pointwise = nn.Conv1d(channels, channels, kernel_size=1)
        self.norm = nn.GroupNorm(num_groups=min(32, channels), num_channels=channels)
        self.activation = nn.SiLU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, C, T]
        residual = x
        h = nn.functional.pad(x, (self.left_pad, 0))  # left-pad only -- causal
        h = self.depthwise(h)
        h = self.pointwise(h)
        h = self.norm(h)
        h = self.activation(h)
        h = self.dropout(h)
        return residual + h


class MixtureAwareEnrollmentAdapter(nn.Module):
    """V3C-MA0 (mixture-aware enrollment conditioning, stage 0). A genuinely
    separate module from the unused `MixtureAwareAdapter` scaffold above
    (that class is dead code, never imported by MentraWearNet -- this is the
    real, wired-in implementation, spec'd precisely by the V3C-MA0 brief).

    Modifies the STATIC enrollment embedding `e` [B, E] into a PER-FRAME
    embedding `e_t` [B, T, E] before it enters the existing
    enrollment_projection -> FiLM path, using the mixture's own per-frame
    representation `m_t` [B, D, T] at tap R1 (post frame_projection, PRE-FiLM
    -- critically NOT post-FiLM, which would already be conditioned on `e`
    and make this circular).

    m'_t = Linear(D -> E)(m_t)                              # project into e's space
    x_t  = [e, m'_t, e * m'_t, |e - m'_t|]                  # [B, T, 4E]
    h_t  = GELU(Linear(4E -> H)(x_t))
    delta_t = Linear(H -> E)(h_t)
    gate_t  = sigmoid(Linear(H -> E)(h_t))
    e_t  = e + alpha * gate_t * delta_t                     # alpha: trainable scalar, init 0

    alpha initialized at EXACTLY 0.0 so e_t == e at step 0 (verified
    numerically by the training driver, not just by construction) --
    gradients still flow to alpha (dL/dalpha = sum(gate*delta * dL/de_t))
    even though alpha itself starts at 0, so training can move it.

    STATIC_ADAPTER control (zero_mixture=True): m'_t is forced to exactly
    zero AFTER the m_proj Linear computes it (not before) -- so m_proj's own
    parameters exist, keep their random init, and have the same shape/count
    as the mixture-aware branch, but receive zero gradient (the module can
    only ever learn e -> e_t, a residual transform of the static embedding
    alone, never anything mixture-dependent). x_t effectively becomes
    [e, 0, 0, |e|].

    Causal by construction: every layer here (m_proj, fc1, delta_proj,
    gate_proj) is a per-frame Linear applied independently to each of the T
    frames -- no temporal convolution, no attention, no pooling across T.
    e_t at frame t depends only on e (static) and m_t (frame t's own
    already-causal representation), never on any other frame.

    Param budget: <100k target (150k hard cap per spec). Default dims
    (E=256, D=192, H=32) measure to ~99.1k params -- see the assertion in
    __init__.
    """

    def __init__(self, embedding_dim: int = 256, mixture_dim: int = 192,
                 hidden_dim: int = 32, zero_mixture: bool = False,
                 max_params: int = 150_000, target_params: int = 100_000):
        super().__init__()
        self.zero_mixture = zero_mixture
        self.embedding_dim = embedding_dim
        self.mixture_dim = mixture_dim
        self.hidden_dim = hidden_dim

        self.m_proj = nn.Linear(mixture_dim, embedding_dim)
        interaction_dim = embedding_dim * 4  # [e, m', e*m', |e-m'|]
        self.fc1 = nn.Linear(interaction_dim, hidden_dim)
        self.delta_proj = nn.Linear(hidden_dim, embedding_dim)
        self.gate_proj = nn.Linear(hidden_dim, embedding_dim)
        self.alpha = nn.Parameter(torch.tensor(0.0))  # exactly 0 at init -- see docstring

        n_params = sum(p.numel() for p in self.parameters())
        self.n_params = n_params
        assert n_params < max_params, (
            f"MixtureAwareEnrollmentAdapter has {n_params:,} params, over the "
            f"{max_params:,} hard cap -- reduce hidden_dim")
        if n_params >= target_params:
            import warnings
            warnings.warn(f"MixtureAwareEnrollmentAdapter has {n_params:,} params, "
                           f"over the {target_params:,} soft target (still under the "
                           f"{max_params:,} hard cap) -- consider reducing hidden_dim")

    def forward(self, e: torch.Tensor, m_t: torch.Tensor) -> torch.Tensor:
        # e: [B, E] (static enrollment embedding)
        # m_t: [B, D, T] (per-frame mixture projection, tap R1, pre-FiLM)
        # returns (e_t, gate, delta), all [B, T, E] -- gate/delta returned
        # too (not just e_t) so a diagnostics script can report gate
        # mean/std and ||delta_t|| broken out by state/TIR without
        # recomputing the adapter.
        m_t = m_t.transpose(1, 2)  # [B, T, D]
        m_proj = self.m_proj(m_t)  # [B, T, E]
        if self.zero_mixture:
            m_proj = torch.zeros_like(m_proj)  # STATIC_ADAPTER control -- see docstring
        T = m_proj.shape[1]
        e_exp = e.unsqueeze(1).expand(-1, T, -1)  # [B, T, E]
        x = torch.cat([e_exp, m_proj, e_exp * m_proj, (e_exp - m_proj).abs()], dim=-1)  # [B, T, 4E]
        h = nn.functional.gelu(self.fc1(x))
        delta = self.delta_proj(h)      # [B, T, E]
        gate = torch.sigmoid(self.gate_proj(h))  # [B, T, E]
        e_t = e_exp + self.alpha * gate * delta  # [B, T, E]
        return e_t, gate, delta


class MentraWearNet(nn.Module):
    def __init__(
        self,
        projection_dim: int = 192,
        similarity_dim: int = 128,
        tcn_blocks: int = 5,
        tcn_kernel_size: int = 3,
        tcn_dilations: tuple[int, ...] = (1, 2, 4, 8, 16),
        dropout: float = 0.1,
        aux_losses: bool = False,
        adapter_mode: str | None = None,
        adapter_hidden_dim: int = 32,
    ):
        super().__init__()
        assert len(tcn_dilations) == tcn_blocks, "one dilation per TCN block"

        # V2 diagnostics/infra addition (sections 2e/2f) -- OFF by default.
        # MentraWearNet() with no args (exactly how training/train.py's
        # current V1 run constructs the model: `model = MentraWearNet()`)
        # gets aux_losses=False, which means NONE of the new aux_* modules
        # below are even created, and process_with_embedding()'s new branch
        # (also gated on self.aux_losses) never executes -- so the
        # architecture, state_dict keys, param count, and forward() output
        # are all byte-identical to before this change for every existing
        # call site. Only a caller that explicitly passes aux_losses=True
        # (not done anywhere in this repo yet -- a future V2 script would
        # opt in) sees any different behavior.
        self.aux_losses = aux_losses

        self.backbone = SpeakerNetBackbone()  # the ONE shared backbone instance (section 6)
        D = projection_dim

        # Section 8: project 1500 -> D immediately, expensive to carry 1500 channels through the TCN.
        self.frame_projection = nn.Conv1d(self.backbone.frame_channels, D, kernel_size=1)

        # Section 9: project the 256-D wearer embedding into the live feature space.
        self.enrollment_projection = nn.Linear(self.backbone.embedding_dim, D)

        # Section 10: FiLM conditioning.
        self.film = FiLM(embedding_dim=D, feature_dim=D)

        # Section 11: explicit cosine-similarity branch -- don't make the TCN
        # rediscover what plain verification already does well.
        self.frame_speaker_projection = nn.Conv1d(D, similarity_dim, kernel_size=1)
        self.enrollment_similarity_projection = nn.Linear(D, similarity_dim)
        self.similarity_fusion = nn.Conv1d(D + 1, D, kernel_size=1)  # concat(h_conditioned, s_t) -> D

        # Section 12: causal depthwise-separable TCN.
        self.tcn = nn.Sequential(*[
            CausalDepthwiseSeparableBlock(D, tcn_kernel_size, dilation, dropout)
            for dilation in tcn_dilations
        ])

        # Section 14: independent logit heads, no sigmoid inside the model (loss takes logits).
        self.wearer_head = nn.Conv1d(D, 1, kernel_size=1)
        self.environment_head = nn.Conv1d(D, 1, kernel_size=1)

        # Section 2e (V2 infra, opt-in): auxiliary training-only heads, only
        # instantiated when aux_losses=True (default False -- see __init__).
        # Removed at export/deployment; not part of the <10M deployment
        # param budget when unused (they don't exist in the module tree at
        # all unless explicitly requested).
        #   any_speech_head: y_speech = y_wearer OR y_environment, 3rd BCE target
        #   four_state_head: 4-way softmax over {00, 10, 01, 11} (silence /
        #     wearer-only / environment-only / overlap), a 4th auxiliary
        #     training signal alongside (not replacing) the two independent
        #     sigmoid heads above.
        if self.aux_losses:
            self.any_speech_head = nn.Conv1d(D, 1, kernel_size=1)
            self.four_state_head = nn.Conv1d(D, 4, kernel_size=1)

        # V3C-MA0 (mixture-aware enrollment conditioning, stage 0), OFF by
        # default (adapter_mode=None), matching the exact pattern already
        # established by aux_losses/speaker_disc above: with adapter_mode
        # left at its default, self.adapter is never even constructed, and
        # process_with_embedding()'s adapter branch below never executes --
        # so architecture, state_dict keys, param count, and forward()
        # output are all byte-identical to before this addition for every
        # existing call site. adapter_mode="static" builds the
        # STATIC_ADAPTER control (mixture information zeroed inside the
        # adapter -- see MixtureAwareEnrollmentAdapter); adapter_mode="mixture"
        # builds the real MIXTURE_AWARE branch.
        assert adapter_mode in (None, "static", "mixture"), \
            f"adapter_mode must be None/'static'/'mixture', got {adapter_mode!r}"
        self.adapter_mode = adapter_mode
        self.adapter = None
        if adapter_mode is not None:
            self.adapter = MixtureAwareEnrollmentAdapter(
                embedding_dim=self.backbone.embedding_dim,  # E, e.g. 256
                mixture_dim=D,                               # tap R1 dim, e.g. 192
                hidden_dim=adapter_hidden_dim,
                zero_mixture=(adapter_mode == "static"),
            )

        self.projection_dim = D

    def encode_enrollment(self, enrollment_waveform: torch.Tensor, enrollment_lengths: torch.Tensor) -> torch.Tensor:
        """Section 22/23: call this ONCE at enrollment time, cache the
        result, never re-run enrollment audio through the graph per
        live-audio call."""
        return self.backbone.encode_speaker(enrollment_waveform, enrollment_lengths)  # [B, 256]

    def process_with_embedding(self, live_waveform: torch.Tensor, live_lengths: torch.Tensor,
                                wearer_embedding: torch.Tensor) -> dict:
        """Deployment-path forward: live audio + a precomputed 256-D wearer
        embedding (no enrollment audio in this call -- section 23)."""
        frame_features, frame_lengths = self.backbone.encode_frames(live_waveform, live_lengths)  # [B,1500,T]
        h = self.frame_projection(frame_features)  # [B, D, T]  -- tap R1: mixture rep, PRE-FiLM

        # e_proj_static: the ORIGINAL static projection of the raw enrollment
        # embedding, computed exactly as before this addition. Kept
        # separate from any adapter output and used UNCHANGED below for the
        # similarity branch (enrollment_sim_space) -- per spec, "do not
        # touch the similarity branch": the explicit cosine-similarity
        # feature and the speaker-discrimination loss must keep anchoring to
        # the static enrolled-speaker identity, never to a per-frame
        # mixture-adapted one (adapting THAT would risk the similarity
        # metric drifting toward whoever's talking, which is exactly the
        # speaker-drift failure mode this experiment tests for elsewhere).
        e_proj_static = self.enrollment_projection(wearer_embedding)  # [B, D]

        # V3C-MA0: when an adapter is active, ALSO compute a PER-FRAME
        # adapted e_t [B,T,E] from `h` (tap R1, i.e. BEFORE self.film --
        # not the FiLM-conditioned output, which would already depend on
        # the original embedding and make this circular), project it with
        # the SAME enrollment_projection Linear (applied per-frame instead
        # of once), and feed ONLY that into FiLM. The existing FiLM module
        # is reused unchanged (its forward() now generalizes to a [B,D,T]
        # conditioning input -- see FiLM.forward).
        if self.adapter is not None:
            e_t, adapter_gate, adapter_delta = self.adapter(wearer_embedding, h)  # each [B, T, E]
            e_proj_film = self.enrollment_projection(e_t)     # [B, T, D] (same Linear as the static path)
            e_proj_film = e_proj_film.transpose(1, 2)          # [B, D, T]
        else:
            e_proj_film = e_proj_static
        h_conditioned = self.film(h, e_proj_film)  # [B, D, T]

        frame_speaker = self.frame_speaker_projection(h_conditioned)  # [B, E, T]
        enrollment_sim_space = self.enrollment_similarity_projection(e_proj_static)  # [B, E] -- static, untouched
        enrollment_sim_space = nn.functional.normalize(enrollment_sim_space, p=2, dim=-1)
        frame_speaker_norm = nn.functional.normalize(frame_speaker, p=2, dim=1)  # normalize over E
        similarity = torch.einsum("bet,be->bt", frame_speaker_norm, enrollment_sim_space)  # [B, T]
        similarity = similarity.unsqueeze(1)  # [B, 1, T]

        fused = self.similarity_fusion(torch.cat([h_conditioned, similarity], dim=1))  # [B, D, T]
        temporal = self.tcn(fused)  # [B, D, T]

        wearer_logits = self.wearer_head(temporal).squeeze(1)  # [B, T]
        environment_logits = self.environment_head(temporal).squeeze(1)  # [B, T]

        out = {
            "wearer_logits": wearer_logits,
            "environment_logits": environment_logits,
            "frame_lengths": frame_lengths,
            "similarity": similarity.squeeze(1),
        }

        # Sections 2e/2f (V2 infra, opt-in) -- this branch only runs when
        # aux_losses=True, which requires the caller to have explicitly
        # constructed MentraWearNet(aux_losses=True) (the any_speech_head /
        # four_state_head modules referenced below only exist in that case
        # -- see __init__). For every existing call site (aux_losses=False,
        # the default), self.aux_losses is False and this entire block is
        # skipped, so `out` above is returned completely unchanged from
        # before this addition.
        if self.aux_losses:
            out["any_speech_logits"] = self.any_speech_head(temporal).squeeze(1)  # [B, T]
            out["four_state_logits"] = self.four_state_head(temporal)  # [B, 4, T]
            # 2f: expose the frame-level speaker-similarity projection (and
            # the enrollment embedding's projection into that same space) so
            # train.py can add an optional non-overlap speaker-discrimination
            # loss without recomputing them -- these are the SAME tensors
            # already computed above for the existing similarity branch,
            # just also returned rather than only consumed internally.
            out["frame_speaker"] = frame_speaker  # [B, E, T], NOT L2-normalized
            out["frame_speaker_norm"] = frame_speaker_norm  # [B, E, T], L2-normalized over E
            out["enrollment_sim_space"] = enrollment_sim_space  # [B, E], L2-normalized

        # V3C-MA0 diagnostics: expose the adapted embedding (and the static
        # one it was derived from) so a diagnostic script can compute
        # cos(e_t, e), ||delta_t||, gate mean/std, and the speaker-drift
        # test without needing to re-run the adapter separately. Only
        # populated when an adapter is active -- zero effect otherwise.
        if self.adapter is not None:
            out["adapted_embedding"] = e_t  # [B, T, E]
            out["static_embedding"] = wearer_embedding  # [B, E]
            out["adapter_gate"] = adapter_gate  # [B, T, E]
            out["adapter_delta"] = adapter_delta  # [B, T, E]

        return out

    def forward(self, mixture_waveform: torch.Tensor, mixture_lengths: torch.Tensor,
                enrollment_waveform: torch.Tensor, enrollment_lengths: torch.Tensor) -> dict:
        """Training-path forward (section 21): computes enrollment embedding
        every call (needed since enrollment speaker varies per training
        example), unlike the deployment path above."""
        wearer_embedding = self.encode_enrollment(enrollment_waveform, enrollment_lengths)
        return self.process_with_embedding(mixture_waveform, mixture_lengths, wearer_embedding)

    def freeze_backbone(self):
        for p in self.backbone.parameters():
            p.requires_grad = False

    def freeze_all_except_adapter(self):
        """V3C-MA0: backbone, projection, FiLM, TCN, and all existing heads
        (activity/4-state/solo-target/speaker-disc) stay frozen at
        STATIC_BASE's weights -- only self.adapter's own parameters train.
        Requires the model to have been constructed with adapter_mode
        not None."""
        assert self.adapter is not None, "freeze_all_except_adapter requires adapter_mode='static'/'mixture'"
        for p in self.parameters():
            p.requires_grad = False
        for p in self.adapter.parameters():
            p.requires_grad = True

    def unfreeze_upper_backbone(self, num_upper_blocks: int = 1):
        """Unfreezes the last `num_upper_blocks` Jasper blocks of the
        ConvASREncoder (section 27, training stage 2), leaving earlier
        blocks + preprocessor frozen."""
        for p in self.backbone.parameters():
            p.requires_grad = False
        encoder_blocks = list(self.backbone.encoder.encoder)  # nn.Sequential of JasperBlocks
        for block in encoder_blocks[-num_upper_blocks:]:
            for p in block.parameters():
                p.requires_grad = True
        for p in self.backbone.pooling.parameters():
            p.requires_grad = True
        for p in self.backbone.embedding_projection.parameters():
            p.requires_grad = True

    def unfreeze_backbone(self):
        for p in self.backbone.parameters():
            p.requires_grad = True

    def deployment_param_count(self) -> int:
        """Total params actually needed at inference time: the full module
        tree here (backbone already excludes the 7205-way classifier since
        SpeakerNetBackbone never loads model.decoder.final -- see
        speakernet_backbone.py)."""
        return sum(p.numel() for p in self.parameters())


class MixtureAwareAdapter(nn.Module):
    """Section 2g (V2 infra, NOT wired into MentraWearNet's active forward
    path). A genuinely separate, standalone module -- importing or
    instantiating this class has zero effect on MentraWearNet's existing
    behavior; nothing in this file's MentraWearNet class references it.

    Implements the review's spec: given the static enrollment embedding
    `e` [B, embedding_dim] and a per-frame mixture projection `m_t`
    [B, T, mixture_dim], produce an ADAPTIVE per-frame conditioning vector
    (as opposed to MentraWearNet's FiLM, which conditions on the STATIC
    embedding alone, identically for every frame). Both `e` and `m_t` are
    first projected down to a small shared `adapter_dim` before forming the
    interaction features `[e, m_t, e*m_t, |e-m_t|]` (concatenated -> 4x
    adapter_dim), which are passed through a small gated MLP:
        gate(feats) * mlp(feats)
    A future MentraWearNetV2 (not implemented here) could feed this
    adapter's output into an additional FiLM-like conditioning step, in
    place of or alongside the current static-embedding FiLM.

    Param budget: <150k (target from the review). Default dims below
    measure to 125,800 params -- see the assertion in __init__.

    Usage example (standalone, not part of any existing training run):

        adapter = MixtureAwareAdapter(embedding_dim=256, mixture_dim=256)
        e = wearer_embedding                       # [B, 256], static per-utterance
        m_t = some_per_frame_mixture_projection     # [B, T, 256]
        adaptive_cond = adapter(e, m_t)             # [B, T, out_dim]
    """

    def __init__(self, embedding_dim: int = 256, mixture_dim: int = 256,
                 adapter_dim: int = 88, hidden_dim: int = 88, out_dim: int = 112,
                 max_params: int = 150_000):
        super().__init__()
        self.e_proj = nn.Linear(embedding_dim, adapter_dim)
        self.m_proj = nn.Linear(mixture_dim, adapter_dim)
        interaction_dim = adapter_dim * 4  # [e, m, e*m, |e-m|]
        self.fc1 = nn.Linear(interaction_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, out_dim)
        self.gate = nn.Linear(interaction_dim, out_dim)
        self.out_dim = out_dim

        n_params = sum(p.numel() for p in self.parameters())
        assert n_params < max_params, (
            f"MixtureAwareAdapter has {n_params:,} params, over the {max_params:,} budget -- "
            f"reduce adapter_dim/hidden_dim/out_dim")

    def forward(self, e: torch.Tensor, m_t: torch.Tensor) -> torch.Tensor:
        # e: [B, embedding_dim] (static enrollment embedding)
        # m_t: [B, T, mixture_dim] (per-frame mixture projection)
        e_p = self.e_proj(e)              # [B, adapter_dim]
        m_p = self.m_proj(m_t)            # [B, T, adapter_dim]
        e_exp = e_p.unsqueeze(1).expand(-1, m_p.shape[1], -1)  # [B, T, adapter_dim]
        feats = torch.cat([e_exp, m_p, e_exp * m_p, (e_exp - m_p).abs()], dim=-1)  # [B, T, 4*adapter_dim]
        h = nn.functional.silu(self.fc1(feats))
        candidate = self.fc2(h)                      # [B, T, out_dim]
        gate = torch.sigmoid(self.gate(feats))        # [B, T, out_dim]
        return gate * candidate                        # [B, T, out_dim] adaptive conditioning vector
