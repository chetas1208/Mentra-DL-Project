#!/usr/bin/env python3
"""Resolve and archive the official NeMo speakerverification_speakernet
checkpoint (sprint spec section 4).

Uses NeMo's own from_pretrained() to resolve the checkpoint (NGC/HF-backed
cache) rather than guessing a download URL, then copies the cached .nemo
file into models/vendor/nemo/ with a recorded SHA256 so it's reproducible
and gitignored.
"""
import argparse
import hashlib
import json
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path


def sha256_of(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model-name", default="speakerverification_speakernet")
    ap.add_argument("--out-dir", type=Path, default=Path("models/vendor/nemo"))
    ap.add_argument("--force", action="store_true",
                     help="overwrite an existing file even if SHA256 differs")
    args = ap.parse_args()

    try:
        import nemo.collections.asr as nemo_asr
        import nemo
    except ImportError as e:
        print(f"BLOCKED: nemo_toolkit not importable ({e})", file=sys.stderr)
        sys.exit(1)

    print(f"resolving {args.model_name} via NeMo from_pretrained (this downloads on first run)...")
    model = nemo_asr.models.EncDecSpeakerLabelModel.from_pretrained(model_name=args.model_name)

    # NeMo caches the .nemo archive under ~/.cache/torch/NeMo/... ; find it
    # via the model's own restoration path if exposed, else search the cache.
    cache_root = Path.home() / ".cache" / "torch" / "NeMo"
    candidates = sorted(cache_root.rglob(f"*{args.model_name}*.nemo")) if cache_root.exists() else []
    if not candidates:
        # fallback: broader search
        candidates = sorted(cache_root.rglob("*.nemo")) if cache_root.exists() else []
    if not candidates:
        print("BLOCKED: could not locate cached .nemo file after from_pretrained() succeeded. "
              "Model loaded in-memory but source archive path unknown -- check NeMo cache "
              "location for this version.", file=sys.stderr)
        sys.exit(1)

    src = candidates[-1]
    args.out_dir.mkdir(parents=True, exist_ok=True)
    dst = args.out_dir / f"{args.model_name}.nemo"

    new_hash = sha256_of(src)
    if dst.exists() and not args.force:
        existing_hash = sha256_of(dst)
        if existing_hash != new_hash:
            print(f"BLOCKED: {dst} exists with a DIFFERENT sha256 "
                  f"(existing={existing_hash[:16]}... new={new_hash[:16]}...). "
                  f"Refusing to overwrite silently -- pass --force if intentional.",
                  file=sys.stderr)
            sys.exit(1)
        print(f"{dst} already present with matching sha256, nothing to do")
    else:
        shutil.copy2(src, dst)
        print(f"copied {src} -> {dst}")

    n_params = sum(p.numel() for p in model.parameters())
    embedding_dim = None
    try:
        embedding_dim = model.decoder.emb_sizes[-1] if hasattr(model.decoder, "emb_sizes") else None
    except Exception:
        pass

    manifest = {
        "name": args.model_name,
        "provider": "NVIDIA NeMo",
        "task": "speaker verification",
        "format": "nemo",
        "source": f"nemo_asr.models.EncDecSpeakerLabelModel.from_pretrained('{args.model_name}')",
        "nemo_version": nemo.__version__,
        "sha256": new_hash,
        "sample_rate": 16000,
        "parameter_count": n_params,
        "embedding_dimension": embedding_dim,
        "license_status": "NOT_VERIFIED",
        "resolved_at": datetime.now(timezone.utc).isoformat(),
        "local_path": str(dst),
    }
    manifest_dir = Path("models/manifests")
    manifest_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = manifest_dir / "speakernet_nemo.json"
    manifest_path.write_text(json.dumps(manifest, indent=2))
    print(f"wrote manifest: {manifest_path}")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
