#!/usr/bin/env python3
"""Real end-to-end transport test over localhost (sprint spec section 22
"Audio tests" + section 23, the localhost-achievable portion of it -- see
docs/MENTRA_REMOTE_AUDIO.md for what's explicitly NOT covered here, i.e.
real glasses + real Tailscale path, which need hardware this dev
environment doesn't have).

Starts a real MentraRemoteReceiver server, connects a real
WebSocketAudioTransport client, sends deterministic synthetic PCM frames,
and verifies bit-identical arrival with correct sequencing -- this is a
genuine network round trip (two asyncio tasks, real OS sockets, real
WebSocket handshake), not a mock.
"""
import asyncio
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from mentra.audio.frame import AudioFrame, Codec, MessageType
from mentra.audio.transport.websocket_transport import WebSocketAudioTransport
from server.audio.remote_receiver import MentraRemoteReceiver

HOST = "127.0.0.1"
PORT = 19199
N_FRAMES = 200
SAMPLES_PER_FRAME = 160  # 10ms @ 16kHz


def make_deterministic_frame(seq: int) -> AudioFrame:
    payload = bytes((seq * 7 + i) % 256 for i in range(SAMPLES_PER_FRAME * 2))  # deterministic, seq-dependent
    return AudioFrame(
        sequence_number=seq,
        capture_timestamp_ns=time.monotonic_ns(),
        sample_rate=16000, channels=1, bits_per_sample=16,
        payload=payload, codec=Codec.PCM16,
    )


async def main():
    received_frames: list[AudioFrame] = []

    def on_frame(frame: AudioFrame):
        received_frames.append(frame)

    receiver = MentraRemoteReceiver(HOST, PORT, jitter_target_ms=20, jitter_max_ms=80)
    server_task = asyncio.create_task(receiver.serve(on_frame))
    await asyncio.sleep(0.2)  # let the server start listening

    client = WebSocketAudioTransport(f"ws://{HOST}:{PORT}", heartbeat_interval_s=1, dead_connection_s=10)
    await client.connect()
    print(f"client connected, session_id={client.session_id}")

    sent_frames = [make_deterministic_frame(i) for i in range(N_FRAMES)]
    t0 = time.monotonic()
    for f in sent_frames:
        await client.send_frame(f)
    send_duration = time.monotonic() - t0

    # let the jitter buffer flush and the receiver catch up
    await asyncio.sleep(0.5)

    # Close the CLIENT side first so the server's ConnectionClosed handler
    # runs its normal finally-block flush() -- cancelling the server task
    # before the client disconnects races the flush and can drop the last
    # buffered frame (found by this test during development, fixed by
    # reordering here + adding JitterBuffer.flush()).
    await client.close()
    await asyncio.sleep(0.3)

    server_task.cancel()

    print(f"\nsent {len(sent_frames)} frames in {send_duration*1000:.1f}ms")
    print(f"received {len(received_frames)} frames")

    ok = True
    if len(received_frames) != N_FRAMES:
        print(f"FAIL: expected {N_FRAMES} frames, got {len(received_frames)}")
        ok = False
    else:
        seqs = [f.sequence_number for f in received_frames]
        if seqs != list(range(N_FRAMES)):
            print(f"FAIL: sequence order wrong: {seqs[:10]}...")
            ok = False
        else:
            print("PASS: all frames received in correct sequence order")

        bit_identical = all(
            received_frames[i].payload == sent_frames[i].payload
            for i in range(N_FRAMES)
        )
        if bit_identical:
            print("PASS: all payloads bit-identical to what was sent")
        else:
            print("FAIL: payload mismatch detected")
            ok = False

    latencies_ms = [(f.receive_timestamp_ns - f.capture_timestamp_ns) / 1e6 for f in received_frames]
    if latencies_ms:
        latencies_ms.sort()
        p50 = latencies_ms[len(latencies_ms) // 2]
        p95 = latencies_ms[int(len(latencies_ms) * 0.95)]
        print(f"\ncapture->receive latency (localhost, real WS round trip): p50={p50:.2f}ms p95={p95:.2f}ms")

    print(f"\nclient stats: sent={client.stats.frames_sent} bytes_sent={client.stats.bytes_sent}")
    print(f"session count on server: {len(receiver.sessions)}")

    print(f"\nOVERALL: {'PASS' if ok else 'FAIL'}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
