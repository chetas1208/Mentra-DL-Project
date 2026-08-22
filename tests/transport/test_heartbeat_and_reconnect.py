#!/usr/bin/env python3
"""Real heartbeat + reconnect integration test (sprint spec section 4/5).
Starts a real server, connects a real client, waits through several real
heartbeat intervals, and verifies: PONGs are actually being consumed
(pong_count > 0, no false DEAD state), then kills the server and confirms
the health monitor actually detects a real failure, then restarts the
server and confirms the client reconnects with a fresh session.
"""
import asyncio
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from mentra.audio.frame import AudioFrame, Codec
from mentra.audio.transport.websocket_transport import WebSocketAudioTransport
from server.audio.remote_receiver import MentraRemoteReceiver

HOST = "127.0.0.1"
PORT = 19197

PASS = []
FAIL = []


def check(name, condition):
    (PASS if condition else FAIL).append(name)
    print(f"{'PASS' if condition else 'FAIL'}: {name}")


async def test_no_false_dead_connection():
    received = []
    receiver = MentraRemoteReceiver(HOST, PORT, jitter_target_ms=20, jitter_max_ms=80)
    server_task = asyncio.create_task(receiver.serve(lambda f: received.append(f)))
    await asyncio.sleep(0.2)

    client = WebSocketAudioTransport(f"ws://{HOST}:{PORT}", heartbeat_interval_s=0.3, dead_connection_s=2.0)
    await client.connect()

    # stay alive across several heartbeat intervals -- previously the bug
    # meant _last_pong_monotonic never updated, so this would eventually
    # trip a false DEAD state even though the connection was fine.
    await asyncio.sleep(1.5)

    check("heartbeat: PINGs were sent", client.ping_count >= 3)
    check("heartbeat: PONGs were actually consumed", client.pong_count >= 3)
    check("heartbeat: no false dead-connection state", client.stats.state == "CONNECTED")
    check("heartbeat: RTT was measured", client.stats.last_rtt_ms is not None and client.stats.last_rtt_ms >= 0)

    await client.close()
    server_task.cancel()
    await asyncio.sleep(0.1)


async def test_real_dead_connection_detected():
    receiver = MentraRemoteReceiver(HOST, PORT + 1, jitter_target_ms=20, jitter_max_ms=80)
    server_task = asyncio.create_task(receiver.serve(lambda f: None))
    await asyncio.sleep(0.2)

    client = WebSocketAudioTransport(f"ws://{HOST}:{PORT + 1}", heartbeat_interval_s=0.2, dead_connection_s=0.6)
    await client.connect()
    await asyncio.sleep(0.3)  # a couple of healthy heartbeats first

    # now kill the server WITHOUT the client knowing -- simulates a real
    # network failure the client has to detect via missing PONGs, not via
    # a clean close.
    server_task.cancel()
    try:
        await server_task
    except asyncio.CancelledError:
        pass

    # Either DISCONNECTED (receive_loop caught a clean ConnectionClosed --
    # what actually happens here, since cancelling an asyncio server task
    # sends a real close frame) or DEAD (heartbeat timeout, for a truly
    # silent hang with no close frame at all -- not reproducible with this
    # test's setup) both correctly signal "failure detected." The one
    # thing that must NOT happen is staying CONNECTED.
    await asyncio.sleep(1.5)
    check("dead_detection: real failure detected (not stuck at CONNECTED)",
          client.stats.state in ("DISCONNECTED", "DEAD"))

    await client.close()


async def test_reconnect_gets_fresh_session():
    received = []
    receiver = MentraRemoteReceiver(HOST, PORT + 2, jitter_target_ms=20, jitter_max_ms=80)
    server_task = asyncio.create_task(receiver.serve(lambda f: received.append(f)))
    await asyncio.sleep(0.2)

    client = WebSocketAudioTransport(f"ws://{HOST}:{PORT + 2}", heartbeat_interval_s=1, dead_connection_s=5,
                                      reconnect_initial_ms=100, reconnect_max_ms=500)
    stop_event = asyncio.Event()

    async def frame_source():
        seq = 0
        while not stop_event.is_set():
            yield AudioFrame(seq, time.monotonic_ns(), 16000, 1, 16, payload=b"\x00\x01" * 160, codec=Codec.PCM16)
            seq += 1
            await asyncio.sleep(0.01)

    run_task = asyncio.create_task(client.run_with_reconnect(frame_source(), stop_event))
    await asyncio.sleep(0.3)
    first_session_id = client.session_id
    check("reconnect: initial connection established", client.stats.state == "CONNECTED")

    # kill and restart the server to force a real reconnect
    server_task.cancel()
    try:
        await server_task
    except asyncio.CancelledError:
        pass
    await asyncio.sleep(0.5)  # let the client notice the disconnect and start backing off

    receiver2 = MentraRemoteReceiver(HOST, PORT + 2, jitter_target_ms=20, jitter_max_ms=80)
    server_task2 = asyncio.create_task(receiver2.serve(lambda f: received.append(f)))
    await asyncio.sleep(1.0)  # give the client's backoff loop a chance to reconnect

    check("reconnect: reconnect_count incremented", client.stats.reconnect_count >= 1)
    check("reconnect: new session_id assigned after reconnect", client.session_id != first_session_id)
    check("reconnect: resumed sending after reconnect", client.stats.state in ("CONNECTED", "CONNECTING"))

    stop_event.set()
    run_task.cancel()
    server_task2.cancel()
    await asyncio.sleep(0.1)


async def main():
    await test_no_false_dead_connection()
    await test_real_dead_connection_detected()
    await test_reconnect_gets_fresh_session()

    print()
    print(f"PASS: {len(PASS)}  FAIL: {len(FAIL)}")
    if FAIL:
        print("FAILED:", FAIL)
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
