#!/usr/bin/env python3
"""Build deterministic FFHQ ECA assets for the Eigenfaces demo.

ECA (opposite of PCA, in the sense of maximally *de-concentrating* the
per-coefficient variance instead of concentrating it): after fitting PCA to
get eigenvectors U and coefficients y_pca = U^T x, additionally left-multiply
by a normalized (orthogonal) Hadamard matrix H, giving basis U' = U @ H and
coefficients y_eca = H^T U^T x. Reconstruction is unaffected (H is
orthogonal and every component is kept), so this only changes how the same
information is spread across the basis/coefficients -- see the U -> UH swap
right after the PCA fit below.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import html
import io
import json
import math
import random
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import requests
import pyarrow.ipc as arrow_ipc
from PIL import Image, ImageOps
from sklearn.decomposition import PCA

SEED = 20260902
SIZE = 128
SAMPLE_COUNT = 5000
DEFAULT_COMPONENTS = 512
# Must be a power of 2: the U -> UH (ECA) transform below needs an exact
# Hadamard matrix of this order. This also means the demo now always
# reconstructs from the full (Hadamard-rotated) basis by default, since
# DEFAULT_COMPONENTS == MAX_COMPONENTS.
MAX_COMPONENTS = 512
EXPOSED_COMPONENTS = 24
CHECKPOINTS = (128, 256, 512)
PORTRAIT_LANDMARKS = {
    "eyeLeft": [423.0, 449.0],
    "eyeRight": [586.0, 465.0],
    "mouthLeft": [409.0, 599.0],
    "mouthRight": [574.0, 612.0],
}


def md5_file(path: Path) -> str:
    digest = hashlib.md5()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def download_one(record: dict, cache: Path) -> Path:
    target = cache / record["thumbnail"]["file_path"]
    expected = record["thumbnail"]["file_md5"]
    if target.exists() and md5_file(target) == expected:
        return target

    target.parent.mkdir(parents=True, exist_ok=True)
    url = record["thumbnail"]["file_url"]
    with requests.Session() as session:
        for attempt in range(4):
            response = session.get(url, timeout=90)
            response.raise_for_status()
            payload = response.content
            if len(payload) < 8192:
                text = payload.decode("utf-8", errors="ignore")
                links = [html.unescape(link) for link in text.split('"') if "export=download" in link]
                if len(links) == 1:
                    url = requests.compat.urljoin(url, links[0])
                    continue
            target.write_bytes(payload)
            if md5_file(target) == expected:
                return target
            target.unlink(missing_ok=True)
            if attempt == 3:
                raise RuntimeError(f"Checksum mismatch for {target.name}")
    raise RuntimeError(f"Could not download {target.name}")


def download_hf_subset(selected_ids: list[str], cache: Path) -> dict[str, Path]:
    """Download only selected rows from an FFHQ 128 mirror when Drive is unavailable."""
    selected_ints = {int(key) for key in selected_ids}
    block_offsets = sorted({(index // 100) * 100 for index in selected_ints})
    row_urls: dict[int, str] = {}

    def fetch_block(offset: int) -> list[tuple[int, str]]:
        for attempt in range(8):
            response = requests.get(
                "https://datasets-server.huggingface.co/rows",
                params={
                    "dataset": "luethan2025/FFHQ-128x128",
                    "config": "default",
                    "split": "train",
                    "offset": offset,
                    "length": 100,
                },
                timeout=90,
            )
            if response.status_code == 429:
                time.sleep(min(30, 1.5 * (attempt + 1)))
                continue
            response.raise_for_status()
            return [
                (int(item["row_idx"]), item["row"]["image"]["src"])
                for item in response.json()["rows"]
                if int(item["row_idx"]) in selected_ints
            ]
        raise RuntimeError(f"Mirror API remained rate limited for block {offset}")

    print(f"Resolving {len(block_offsets)} mirror row blocks…", flush=True)
    with ThreadPoolExecutor(max_workers=4) as executor:
        futures = [executor.submit(fetch_block, offset) for offset in block_offsets]
        for index, future in enumerate(as_completed(futures), start=1):
            row_urls.update(future.result())
            if index % 50 == 0:
                print(f"  blocks {index}/{len(block_offsets)}", flush=True)

    if len(row_urls) != len(selected_ids):
        raise RuntimeError(f"Resolved {len(row_urls)} of {len(selected_ids)} selected rows")

    mirror_dir = cache / "hf-thumbnails"
    mirror_dir.mkdir(parents=True, exist_ok=True)

    def fetch_image(index: int) -> tuple[int, Path]:
        target = mirror_dir / f"{index:05d}.jpg"
        if target.exists():
            try:
                with Image.open(target) as image:
                    if image.size == (SIZE, SIZE):
                        return index, target
            except Exception:
                target.unlink(missing_ok=True)
        for attempt in range(4):
            response = requests.get(row_urls[index], timeout=90)
            if response.ok:
                target.write_bytes(response.content)
                try:
                    with Image.open(target) as image:
                        if image.size == (SIZE, SIZE):
                            return index, target
                except Exception:
                    pass
                target.unlink(missing_ok=True)
            if attempt == 3:
                response.raise_for_status()
        raise RuntimeError(f"Could not download mirror row {index}")

    downloaded: dict[str, Path] = {}
    print(f"Downloading/validating {len(selected_ids)} mirror thumbnails…", flush=True)
    with ThreadPoolExecutor(max_workers=32) as executor:
        futures = [executor.submit(fetch_image, int(key)) for key in selected_ids]
        for count, future in enumerate(as_completed(futures), start=1):
            index, path = future.result()
            downloaded[str(index)] = path
            if count % 250 == 0:
                print(f"  {count}/{len(selected_ids)}", flush=True)
    return downloaded


def load_gray(path: Path) -> np.ndarray:
    with Image.open(path) as source:
        rgb = np.asarray(source.convert("RGB"), dtype=np.float32) / 255.0
    gray = rgb[..., 0] * 0.2126 + rgb[..., 1] * 0.7152 + rgb[..., 2] * 0.0722
    return gray.astype(np.float32)


def load_arrow_matrix(arrow_dir: Path, selected_ids: list[str]) -> np.ndarray:
    selected = {int(key): position for position, key in enumerate(selected_ids)}
    matrix = np.empty((len(selected_ids), SIZE * SIZE), dtype=np.float32)
    found = 0
    global_index = 0
    files = sorted(arrow_dir.glob("*.arrow"))
    if not files:
        raise RuntimeError(f"No Arrow files found in {arrow_dir}")
    print(f"Reading selected thumbnails from {len(files)} Arrow files…", flush=True)
    for path in files:
        with path.open("rb") as handle:
            reader = arrow_ipc.open_stream(handle)
            for batch in reader:
                image_column = batch.column(0)
                for row in range(batch.num_rows):
                    position = selected.get(global_index)
                    if position is not None:
                        payload = image_column[row].as_py()["bytes"]
                        with Image.open(io.BytesIO(payload)) as image:
                            rgb = np.asarray(image.convert("RGB"), dtype=np.float32) / 255.0
                        gray = rgb[..., 0] * 0.2126 + rgb[..., 1] * 0.7152 + rgb[..., 2] * 0.0722
                        matrix[position] = gray.reshape(-1)
                        found += 1
                    global_index += 1
    if found != len(selected_ids):
        raise RuntimeError(f"Read {found} of {len(selected_ids)} selected rows")
    return matrix


def save_gray(path: Path, values: np.ndarray) -> None:
    pixels = np.clip(values, 0.0, 1.0)
    image = Image.fromarray(np.rint(pixels * 255).astype(np.uint8))
    image.save(path, optimize=True)


def align_portrait(source_path: Path, cache: Path) -> tuple[np.ndarray, np.ndarray]:
    with Image.open(source_path) as source:
        image = ImageOps.exif_transpose(source).convert("RGB")

    # Manually verified landmarks for this fixed supplied portrait. Coordinates
    # follow FFHQ's 68-point convention: eye centers and mouth corners.
    eye_left = np.array(PORTRAIT_LANDMARKS["eyeLeft"], dtype=np.float64)
    eye_right = np.array(PORTRAIT_LANDMARKS["eyeRight"], dtype=np.float64)
    mouth_left = np.array(PORTRAIT_LANDMARKS["mouthLeft"], dtype=np.float64)
    mouth_right = np.array(PORTRAIT_LANDMARKS["mouthRight"], dtype=np.float64)

    eye_avg = (eye_left + eye_right) * 0.5
    eye_to_eye = eye_right - eye_left
    mouth_avg = (mouth_left + mouth_right) * 0.5
    eye_to_mouth = mouth_avg - eye_avg

    # This is the quadrilateral construction from FFHQ's published align_face.py.
    # It standardizes roll, scale, and vertical placement jointly rather than
    # rotating and cropping as two unrelated operations.
    x_axis = eye_to_eye - np.flipud(eye_to_mouth) * np.array([-1.0, 1.0])
    x_axis /= np.hypot(*x_axis)
    x_axis *= max(np.hypot(*eye_to_eye) * 2.0, np.hypot(*eye_to_mouth) * 1.8)
    y_axis = np.flipud(x_axis) * np.array([-1.0, 1.0])
    center = eye_avg + eye_to_mouth * 0.1
    quad = np.stack(
        [
            center - x_axis - y_axis,
            center - x_axis + y_axis,
            center + x_axis + y_axis,
            center + x_axis - y_axis,
        ]
    )

    # Transform at high resolution before the final Lanczos reduction so the
    # 128px training convention does not introduce avoidable resampling noise.
    transform_size = 1024
    image = image.transform(
        (transform_size, transform_size),
        Image.Transform.QUAD,
        tuple((quad + 0.5).reshape(-1)),
        Image.Resampling.BICUBIC,
    ).resize((SIZE, SIZE), Image.Resampling.LANCZOS)
    aligned_path = cache / "aligned-input.png"
    aligned_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(aligned_path, optimize=True)
    rgb = np.asarray(image, dtype=np.float32) / 255.0
    grayscale = (
        rgb[..., 0] * 0.2126 + rgb[..., 1] * 0.7152 + rgb[..., 2] * 0.0722
    ).astype(np.float32)
    return grayscale, quad


def write_float32(path: Path, values: np.ndarray) -> str:
    contiguous = np.asarray(values, dtype="<f4").reshape(-1)
    path.write_bytes(contiguous.tobytes())
    return hashlib.sha256(path.read_bytes()).hexdigest()


def hadamard(n: int) -> np.ndarray:
    """Exact Sylvester-construction Hadamard matrix of order n (a power of 2).
    Symmetric by construction (H^T == H), which is used below to fold the
    U -> UH substitution into a single left-multiply of both the basis and
    the coefficients."""
    assert (n & (n - 1) == 0) and n > 0, "Hadamard order must be a power of 2"
    matrix = np.array([[1.0]])
    while matrix.shape[0] < n:
        matrix = np.block([[matrix, matrix], [matrix, -matrix]])
    return matrix


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--portrait", type=Path, required=True)
    parser.add_argument("--cache", type=Path, default=Path(".cache/eigenfaces"))
    parser.add_argument("--output", type=Path, default=Path("public/eigenfaces"))
    parser.add_argument("--hf-mirror", action="store_true")
    parser.add_argument("--arrow-dir", type=Path)
    args = parser.parse_args()

    args.cache.mkdir(parents=True, exist_ok=True)
    args.output.mkdir(parents=True, exist_ok=True)

    print("Loading FFHQ metadata…", flush=True)
    records = json.loads(args.metadata.read_text())
    training_ids = sorted(
        (key for key, value in records.items() if value["category"] == "training"),
        key=int,
    )
    selected_ids = sorted(random.Random(SEED).sample(training_ids, SAMPLE_COUNT), key=int)
    selected = [records[key] for key in selected_ids]

    if args.arrow_dir:
        matrix = load_arrow_matrix(args.arrow_dir, selected_ids)
    elif args.hf_mirror:
        downloaded = download_hf_subset(selected_ids, args.cache)
    else:
        print(f"Downloading/validating {SAMPLE_COUNT} thumbnails…", flush=True)
        downloaded: dict[str, Path] = {}
        with ThreadPoolExecutor(max_workers=24) as executor:
            futures = {
                executor.submit(download_one, record, args.cache): key
                for key, record in zip(selected_ids, selected)
            }
            for index, future in enumerate(as_completed(futures), start=1):
                key = futures[future]
                downloaded[key] = future.result()
                if index % 250 == 0:
                    print(f"  {index}/{SAMPLE_COUNT}", flush=True)

    if not args.arrow_dir:
        print("Decoding grayscale training matrix…", flush=True)
        matrix = np.empty((SAMPLE_COUNT, SIZE * SIZE), dtype=np.float32)
        for index, key in enumerate(selected_ids):
            matrix[index] = load_gray(downloaded[key]).reshape(-1)

    print(f"Fitting {MAX_COMPONENTS}-component randomized PCA…", flush=True)
    pca = PCA(
        n_components=MAX_COMPONENTS,
        svd_solver="randomized",
        random_state=SEED,
        iterated_power=4,
    )
    pca.fit(matrix)
    components = pca.components_.astype(np.float32)  # U^T: row i is eigenvector u_i

    for index in range(components.shape[0]):
        pivot = int(np.argmax(np.abs(components[index])))
        if components[index, pivot] < 0:
            components[index] *= -1

    portrait_image, alignment_quad = align_portrait(args.portrait, args.cache)
    portrait = portrait_image.reshape(-1)
    centered = portrait - pca.mean_.astype(np.float32)
    weights = components @ centered  # y_pca = U^T x

    # --- ECA: U -> U @ H -------------------------------------------------
    # H is a normalized (orthogonal, and for the Sylvester construction
    # symmetric: H^T == H) Hadamard matrix. Substituting U with UH means:
    #   (UH)^T           = H^T U^T        = H @ components
    #   y_eca = H^T U^T x = H^T y_pca      = H @ weights
    # Reconstruction is unaffected since every component is kept and H is
    # orthogonal: (UH) @ (H @ weights) = U @ (H @ H) @ weights = U @ weights.
    print(f"Applying ECA: U -> UH with a {MAX_COMPONENTS}x{MAX_COMPONENTS} Hadamard…", flush=True)
    H = hadamard(MAX_COMPONENTS) / math.sqrt(MAX_COMPONENTS)
    components = (H @ components).astype(np.float32)
    weights = (H @ weights).astype(np.float32)

    # A normalized Hadamard block exactly equalizes output variance: for
    # per-axis PCA variances var_1..var_k, every one of the k mixed outputs
    # ends up with the same variance (1/k) * sum(var_i). So unlike PCA's
    # per-component eigenvalue/explained-variance curve, every ECA
    # coefficient shares one common variance and the cumulative-variance
    # curve below is linear rather than concentrated.
    total_variance = float(pca.explained_variance_[0] / pca.explained_variance_ratio_[0])
    avg_eigenvalue = float(pca.explained_variance_[:MAX_COMPONENTS].mean())
    avg_explained_variance = avg_eigenvalue / total_variance
    cumulative = avg_explained_variance * np.arange(1, MAX_COMPONENTS + 1)

    baseline = (
        pca.mean_.astype(np.float32)
        + components[:DEFAULT_COMPONENTS].T @ weights[:DEFAULT_COMPONENTS]
    )

    # Cache the complete fitted basis and projection offline. The browser uses a
    # compact table of quantized prefix reconstructions so changing k only selects
    # a precomputed projection; PCA is never refit during interaction.
    write_float32(args.cache / f"components-{MAX_COMPONENTS}.f32", components)
    write_float32(args.cache / f"portrait-weights-{MAX_COMPONENTS}.f32", weights)
    prefix_reconstructions = np.empty(
        (MAX_COMPONENTS, SIZE * SIZE), dtype=np.uint8
    )
    running = pca.mean_.astype(np.float32).copy()
    for index in range(MAX_COMPONENTS):
        running += components[index] * weights[index]
        prefix_reconstructions[index] = np.rint(
            np.clip(running, 0.0, 1.0) * 255.0
        ).astype(np.uint8)
    prefix_deltas = np.empty_like(prefix_reconstructions)
    prefix_deltas[0] = prefix_reconstructions[0]
    prefix_deltas[1:] = (
        prefix_reconstructions[1:].astype(np.int16)
        - prefix_reconstructions[:-1].astype(np.int16)
    ) % 256
    prefix_path = args.output / "prefix-reconstructions.delta.bin"
    prefix_path.write_bytes(gzip.compress(prefix_deltas.tobytes(), compresslevel=9, mtime=0))
    (args.output / "prefix-reconstructions.u8").unlink(missing_ok=True)
    (args.output / "prefix-reconstructions.delta.u8.gz").unlink(missing_ok=True)
    prefix_sha = hashlib.sha256(prefix_path.read_bytes()).hexdigest()

    save_gray(args.output / "mean.png", pca.mean_.reshape(SIZE, SIZE))
    save_gray(args.output / "reconstruction.png", baseline.reshape(SIZE, SIZE))

    baseline_sha = write_float32(args.output / "baseline.f32", baseline)
    component_records = []
    for index in range(EXPOSED_COMPONENTS):
        vector = components[index]
        amplitude = float(np.max(np.abs(vector))) or 1.0
        thumbnail = 0.5 + 0.5 * vector.reshape(SIZE, SIZE) / amplitude
        thumbnail_path = args.output / f"pc-{index + 1:02d}.png"
        vector_path = args.output / f"pc-{index + 1:02d}.f32"
        save_gray(thumbnail_path, thumbnail)
        component_records.append(
            {
                "index": index + 1,
                # every ECA coefficient shares the same (Hadamard-equalized) variance
                "eigenvalue": avg_eigenvalue,
                "explainedVariance": avg_explained_variance,
                "baselineWeight": float(weights[index]),
                "baselineZ": float(weights[index] / math.sqrt(avg_eigenvalue)),
                "vector": f"/eigenfaces/{vector_path.name}",
                "vectorSha256": write_float32(vector_path, vector),
                "thumbnail": f"/eigenfaces/{thumbnail_path.name}",
            }
        )

    provenance = []
    for key, record in zip(selected_ids, selected):
        metadata = record["metadata"]
        provenance.append(
            {
                "id": int(key),
                "thumbnailMd5": record["thumbnail"]["file_md5"],
                "author": metadata.get("author", ""),
                "sourceUrl": metadata.get("photo_url", ""),
                "license": metadata.get("license", ""),
                "licenseUrl": metadata.get("license_url", ""),
            }
        )
    (args.cache / "provenance.json").write_text(json.dumps(provenance, indent=2))

    manifest = {
        "schemaVersion": 1,
        "dataset": "Flickr-Faces-HQ (FFHQ)",
        "datasetUrl": "https://github.com/NVlabs/ffhq-dataset",
        "seed": SEED,
        "sampleCount": SAMPLE_COUNT,
        "width": SIZE,
        "height": SIZE,
        "flatteningOrder": "row-major",
        "grayscale": "sRGB luminance 0.2126R + 0.7152G + 0.0722B",
        "kFull": DEFAULT_COMPONENTS,
        "defaultDimensions": DEFAULT_COMPONENTS,
        "maxDimensions": MAX_COMPONENTS,
        "exposedComponentCount": EXPOSED_COMPONENTS,
        "explainedVariance": {
            str(checkpoint): float(cumulative[checkpoint - 1])
            for checkpoint in CHECKPOINTS
        },
        "cumulativeExplainedVariance": [float(value) for value in cumulative],
        "baseline": "/eigenfaces/baseline.f32",
        "baselineSha256": baseline_sha,
        "prefixReconstructions": f"/eigenfaces/{prefix_path.name}",
        "prefixReconstructionsSha256": prefix_sha,
        "prefixEncoding": "gzip-delta-uint8-clamped-row-major",
        "mean": "/eigenfaces/mean.png",
        "reconstruction": "/eigenfaces/reconstruction.png",
        "components": component_records,
        "portraitAlignment": {
            "method": "FFHQ eye/mouth-oriented quadrilateral",
            "landmarks": PORTRAIT_LANDMARKS,
            "quad": alignment_quad.tolist(),
            "sourceSize": [914, 1003],
        },
    }
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(
        f"Wrote assets with default={DEFAULT_COMPONENTS}, max={MAX_COMPONENTS}; "
        f"variance@{MAX_COMPONENTS}={cumulative[-1]:.4f}",
        flush=True,
    )


if __name__ == "__main__":
    main()
