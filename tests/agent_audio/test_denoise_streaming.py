from __future__ import annotations

import numpy as np

from evaluation.agent_audio.denoise import Denoiser


class _FakeRNNoise:
    FRAME_SIZE = 4

    def __init__(self):
        self.created = 0
        self.destroyed = 0
        self.frames = 0

    def create(self):
        self.created += 1
        return object()

    def destroy(self, state):
        assert state is not None
        self.destroyed += 1

    def process_mono_frame(self, state, frame):
        assert state is not None
        self.frames += 1
        return np.asarray(frame, dtype=np.int16), 0.5


def test_denoiser_preserves_rnnoise_state_until_explicit_reset():
    fake = _FakeRNNoise()
    denoiser = Denoiser(sr=48_000)
    denoiser.ll = fake
    denoiser.available = True

    denoiser.process(np.ones(4, dtype=np.float32) * 0.1)
    denoiser.process(np.ones(4, dtype=np.float32) * 0.2)

    assert fake.created == 1
    assert fake.destroyed == 0
    assert fake.frames == 2

    denoiser.reset()
    assert fake.destroyed == 1
