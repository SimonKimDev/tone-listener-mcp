#!/usr/bin/env python3
"""Reproducibly derive the bundled lead-1/lead-2 DIs from Guitar-TECHS P3_music.zip.

Usage: python3 scripts/prepare_lead_dis.py /path/to/P3_music.zip [--check]

The zip is CC BY 4.0 (Zenodo record 14963133). Processing is deliberately minimal:
verify hashes, verify exact dual-mono, keep channel 0, trim, polyphase-resample
48 kHz -> 44.1 kHz, write 24-bit PCM. No normalization or gain is applied to the
body; only 10 ms raised-cosine fades at the two cut points avoid edge clicks.
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import sys
import zipfile
from pathlib import Path

import numpy as np
import soundfile as sf
from scipy.signal import resample_poly

DI_ROOT = Path(__file__).resolve().parents[1] / "src" / "tone_listener_mcp" / "workbench" / "assets" / "di"
ZIP_SHA256 = "033489e22600751fb5a1633e7d856b901c6782e0486fa02135e830780d9dbfe2"
SOURCE_RATE = 48_000
TARGET_RATE = 44_100
FADE_SECONDS = 0.010
# (bundled name, member, member sha256, start s, end s) in source-file time.
LEADS = (
    ("lead-1", "P3_music/audio/directinput/directinput_08.wav",
     "f58cd8dcd68eeec7c9e58839e7161671ba3af9963957e35f1f04a1f2644d3ff3", 8.60, 23.80),
    ("lead-2", "P3_music/audio/directinput/directinput_10.wav",
     "d50cc046c4d1d39899c534738377b9d0c5d5396bbed980a029c61b7e4a77d6b8", 3.86, 16.29),
)


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _nearest_zero_crossing(x: np.ndarray, index: int, radius: int = 480) -> int:
    lo, hi = max(1, index - radius), min(len(x) - 1, index + radius)
    sign = np.signbit(x[lo - 1:hi - 1]) != np.signbit(x[lo:hi])
    hits = np.flatnonzero(sign) + lo
    return int(hits[np.argmin(np.abs(hits - index))]) if hits.size else index


def derive(zip_path: Path) -> dict[str, tuple[np.ndarray, dict]]:
    raw = zip_path.read_bytes()
    if _sha(raw) != ZIP_SHA256:
        raise SystemExit("P3_music.zip sha256 does not match the pinned Zenodo file.")
    out = {}
    with zipfile.ZipFile(io.BytesIO(raw)) as archive:
        for name, member, digest, start, end in LEADS:
            data = archive.read(member)
            if _sha(data) != digest:
                raise SystemExit(f"{member} sha256 mismatch.")
            audio, rate = sf.read(io.BytesIO(data), dtype="float64", always_2d=True)
            if rate != SOURCE_RATE or audio.shape[1] != 2 or not np.array_equal(audio[:, 0], audio[:, 1]):
                raise SystemExit(f"{member} is not exact dual-mono 48 kHz.")
            mono = audio[:, 0]
            a = _nearest_zero_crossing(mono, round(start * rate))
            b = _nearest_zero_crossing(mono, round(end * rate))
            segment = resample_poly(mono[a:b], TARGET_RATE // 300, SOURCE_RATE // 300)
            fade = round(FADE_SECONDS * TARGET_RATE)
            ramp = 0.5 - 0.5 * np.cos(np.linspace(0, np.pi, fade))
            segment[:fade] *= ramp
            segment[-fade:] *= ramp[::-1]
            if np.max(np.abs(segment)) >= 1.0:
                raise SystemExit(f"{name} would clip; refusing to apply gain.")
            out[name] = (segment, {
                "file": f"{name}.wav", "upstream_member": member, "upstream_sha256": digest,
                "trim_start_seconds": round(a / rate, 6), "trim_end_seconds": round(b / rate, 6),
                "trim_start_sample_48k": a, "trim_end_sample_48k": b,
            })
    return out


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("zip", type=Path)
    parser.add_argument("--check", action="store_true", help="verify bundled files instead of writing")
    args = parser.parse_args()
    provenance = json.loads((DI_ROOT / "provenance.json").read_text())
    recorded = {item["file"]: item for source in provenance.get("additional_sources", [])
                for item in source["samples"]}
    for name, (segment, meta) in derive(args.zip).items():
        buffer = io.BytesIO()
        sf.write(buffer, segment, TARGET_RATE, subtype="PCM_24", format="WAV")
        digest = _sha(buffer.getvalue())
        target = DI_ROOT / meta["file"]
        if args.check:
            if digest != recorded.get(meta["file"], {}).get("sha256") or _sha(target.read_bytes()) != digest:
                print(f"{name}: MISMATCH", file=sys.stderr)
                return 1
        else:
            target.write_bytes(buffer.getvalue())
        print(json.dumps({**meta, "sha256": digest, "seconds": round(len(segment) / TARGET_RATE, 3)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
