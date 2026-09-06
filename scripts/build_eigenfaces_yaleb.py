#!/usr/bin/env python3
"""Build deterministic Extended Yale B PCA + ECA assets for the GhostFaces
demo, side by side.

This is a local-data stand-in for build_eigenfaces.py: same PCA -> ECA math,
but fits on the local Extended Yale B database instead of downloading FFHQ,
and reconstructs a held-out Yale B image instead of requiring a personal
portrait + manually-picked face landmarks.

Unlike the single-basis build_eigenfaces.py, this script keeps BOTH the raw
PCA basis/coefficients and the Hadamard-rotated ECA ones, and writes a
prefix-reconstruction table for each, so the frontend can drive two
reconstructions -- one per method -- off a single shared "how many
components" slider and show both convergence curves at once:

  PCA: y_pca = U^T x                (natural order = eigenvalue-sorted)
  ECA: y_eca = H^T U^T x = H @ y_pca  (H = normalized Hadamard matrix)

Reconstruction is unaffected by the U -> UH substitution (H is orthogonal
and every component is kept), so both prefix tables converge to the same
final image at k = MAX_COMPONENTS -- they only differ in how fast they get
there, which is the point of the comparison.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import math
import random
import re
from pathlib import Path

import numpy as np
from PIL import Image
from sklearn.decomposition import PCA

SEED = 20260902
SIZE = 128
# Initial position of the shared "components used" slider. Deliberately
# small: at this k, PCA has already captured most of its target's identity
# while ECA (equal variance per coefficient by construction) has barely
# moved off the mean -- the comparison the slider exists to show.
DEFAULT_COMPONENTS = 16
# Must be a power of 2: the U -> UH (ECA) transform needs an exact Hadamard
# matrix of this order. Also must stay below the ~2412 non-trivial rank of
# the centered Yale B data (2413 training images minus 1 for centering) --
# matches the 2048 used earlier in eca_eigenfaces.py / oca_comparison.py for
# this same reason.
MAX_COMPONENTS = 2048
EXPOSED_COMPONENTS = 24

FNAME_RE = re.compile(r"^(yaleB\d{2})_(P\d{2}A[+-]\d{3}E[+-]\d{2})\.pgm$")


def list_frontal_images(raw_dir: Path) -> list[tuple[str, str, Path]]:
    """Usable frontal Extended Yale B images: excludes 'Ambient' shots and
    any file the original distributors flagged corrupted (kept as
    *.pgm.bad, so a plain *.pgm glob already skips them)."""
    items = []
    for subj_dir in sorted(raw_dir.iterdir()):
        if not subj_dir.is_dir():
            continue
        for path in sorted(subj_dir.iterdir()):
            m = FNAME_RE.match(path.name)
            if not m:
                continue
            items.append((m.group(1), m.group(2), path))
    items.sort(key=lambda t: (t[0], t[1]))
    return items


def load_gray_resized(path: Path) -> np.ndarray:
    """Load a Yale B pgm and resize to SIZE x SIZE to match the frontend's
    fixed pixel-count assumption (Yale B ships at 192x168)."""
    with Image.open(path) as source:
        image = source.convert("L").resize((SIZE, SIZE), Image.Resampling.LANCZOS)
    return (np.asarray(image, dtype=np.float32) / 255.0).reshape(-1)


def hadamard(n: int) -> np.ndarray:
    """Exact Sylvester-construction Hadamard matrix of order n (a power of
    2). Symmetric by construction (H^T == H), which is used below to fold
    the U -> UH substitution into a single left-multiply of both the basis
    and the coefficients."""
    assert (n & (n - 1) == 0) and n > 0, "Hadamard order must be a power of 2"
    matrix = np.array([[1.0]])
    while matrix.shape[0] < n:
        matrix = np.block([[matrix, matrix], [matrix, -matrix]])
    return matrix


def write_float32(path: Path, values: np.ndarray) -> str:
    contiguous = np.asarray(values, dtype="<f4").reshape(-1)
    path.write_bytes(contiguous.tobytes())
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save_gray(path: Path, values: np.ndarray) -> None:
    pixels = np.clip(values, 0.0, 1.0)
    image = Image.fromarray(np.rint(pixels * 255).astype(np.uint8))
    image.save(path, optimize=True)


def build_prefix_table(mean: np.ndarray, components: np.ndarray, weights: np.ndarray) -> np.ndarray:
    """Progressive reconstruction mean + sum_{i<k} components[i]*weights[i]
    for every k in 1..MAX_COMPONENTS, quantized to uint8. `components` must
    already be in the order this method wants to add terms in (eigenvalue
    order for PCA, natural Hadamard order for ECA)."""
    table = np.empty((MAX_COMPONENTS, SIZE * SIZE), dtype=np.uint8)
    running = mean.copy()
    for index in range(MAX_COMPONENTS):
        running += components[index] * weights[index]
        table[index] = np.rint(np.clip(running, 0.0, 1.0) * 255.0).astype(np.uint8)
    return table


def write_prefix_table(path: Path, table: np.ndarray) -> str:
    deltas = np.empty_like(table)
    deltas[0] = table[0]
    deltas[1:] = (table[1:].astype(np.int16) - table[:-1].astype(np.int16)) % 256
    path.write_bytes(gzip.compress(deltas.tobytes(), compresslevel=9, mtime=0))
    return hashlib.sha256(path.read_bytes()).hexdigest()


def build_method_assets(
    output: Path,
    slug: str,
    label: str,
    mean: np.ndarray,
    components: np.ndarray,
    weights: np.ndarray,
    cumulative: np.ndarray,
    per_component_variance,
) -> dict:
    """Write one method's (PCA or ECA) prefix table + component thumbnails,
    and return its manifest.methods[slug] record."""
    prefix_path = output / f"prefix-reconstructions-{slug}.delta.bin"
    table = build_prefix_table(mean, components, weights)
    prefix_sha = write_prefix_table(prefix_path, table)

    component_records = []
    for index in range(EXPOSED_COMPONENTS):
        vector = components[index]
        amplitude = float(np.max(np.abs(vector))) or 1.0
        thumbnail = 0.5 + 0.5 * vector.reshape(SIZE, SIZE) / amplitude
        thumbnail_path = output / f"pc-{slug}-{index + 1:02d}.png"
        save_gray(thumbnail_path, thumbnail)
        eigenvalue, explained_variance = per_component_variance(index)
        component_records.append(
            {
                "index": index + 1,
                "eigenvalue": eigenvalue,
                "explainedVariance": explained_variance,
                "thumbnail": f"/eigenfaces/{thumbnail_path.name}",
            }
        )

    return {
        "label": label,
        "prefixReconstructions": f"/eigenfaces/{prefix_path.name}",
        "prefixReconstructionsSha256": prefix_sha,
        "cumulativeExplainedVariance": [float(value) for value in cumulative],
        "components": component_records,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-dir", type=Path,
                         default=Path("/data/dwaipayanh/eigenfaces_eca/raw/CroppedYale"))
    parser.add_argument("--cache", type=Path, default=Path(".cache/eigenfaces_yaleb"))
    parser.add_argument("--output", type=Path, default=Path("public/eigenfaces"))
    args = parser.parse_args()

    args.cache.mkdir(parents=True, exist_ok=True)
    args.output.mkdir(parents=True, exist_ok=True)

    print("Listing local Extended Yale B images…", flush=True)
    items = list_frontal_images(args.raw_dir)
    if not items:
        raise RuntimeError(f"No usable Yale B images found under {args.raw_dir}")

    # Hold one image out as the "portrait" to reconstruct, exactly the role
    # FFHQ's build_eigenfaces.py gives a personal photo -- except this one is
    # already part of the public, licensed Yale B corpus, so no upload or
    # manual eye/mouth landmark-picking is needed.
    portrait_subj, portrait_code, portrait_path = random.Random(SEED).choice(items)
    training_items = [item for item in items if item[2] != portrait_path]
    print(f"{len(items)} usable images; held out {portrait_subj}-{portrait_code} as the portrait, "
          f"fitting PCA on the remaining {len(training_items)}.", flush=True)

    print("Loading and resizing training matrix…", flush=True)
    matrix = np.empty((len(training_items), SIZE * SIZE), dtype=np.float32)
    for index, (_, _, path) in enumerate(training_items):
        matrix[index] = load_gray_resized(path)

    print(f"Fitting {MAX_COMPONENTS}-component PCA…", flush=True)
    pca = PCA(
        # "full" (exact SVD), not "randomized": we're extracting ~85% of the
        # data's non-trivial rank (2048 of ~2412), well past the regime where
        # randomized SVD's low-rank approximation stays accurate.
        n_components=MAX_COMPONENTS,
        svd_solver="full",
        random_state=SEED,
    )
    pca.fit(matrix)
    components_pca = pca.components_.astype(np.float32)  # U^T: row i is eigenvector u_i

    for index in range(components_pca.shape[0]):
        pivot = int(np.argmax(np.abs(components_pca[index])))
        if components_pca[index, pivot] < 0:
            components_pca[index] *= -1

    mean = pca.mean_.astype(np.float32)
    portrait = load_gray_resized(portrait_path)
    centered = portrait - mean
    weights_pca = components_pca @ centered  # y_pca = U^T x
    cumulative_pca = np.cumsum(pca.explained_variance_ratio_[:MAX_COMPONENTS])

    # --- ECA: U -> U @ H -------------------------------------------------
    # H is a normalized (orthogonal, and for the Sylvester construction
    # symmetric: H^T == H) Hadamard matrix. Substituting U with UH means:
    #   (UH)^T            = H^T U^T   = H @ components
    #   y_eca = H^T U^T x = H^T y_pca = H @ weights
    # Reconstruction is unaffected since every component is kept and H is
    # orthogonal: (UH) @ (H @ weights) = U @ (H @ H) @ weights = U @ weights.
    print(f"Applying ECA: U -> UH with a {MAX_COMPONENTS}x{MAX_COMPONENTS} Hadamard…", flush=True)
    H = hadamard(MAX_COMPONENTS) / math.sqrt(MAX_COMPONENTS)
    components_eca = (H @ components_pca).astype(np.float32)
    weights_eca = (H @ weights_pca).astype(np.float32)

    # A normalized Hadamard block exactly equalizes output variance: for
    # per-axis PCA variances var_1..var_k, every one of the k mixed outputs
    # ends up with the same variance (1/k) * sum(var_i). So unlike PCA's
    # per-component eigenvalue/explained-variance curve (concentrated, fast
    # convergence), every ECA coefficient shares one common variance and its
    # cumulative-variance curve is linear -- convergence only completes once
    # nearly all components are included.
    total_variance = float(pca.explained_variance_[0] / pca.explained_variance_ratio_[0])
    avg_eigenvalue = float(pca.explained_variance_[:MAX_COMPONENTS].mean())
    avg_explained_variance = avg_eigenvalue / total_variance
    cumulative_eca = avg_explained_variance * np.arange(1, MAX_COMPONENTS + 1)

    print("Writing PCA assets…", flush=True)
    pca_record = build_method_assets(
        args.output, "pca", "Eigenfaces", mean, components_pca, weights_pca, cumulative_pca,
        lambda index: (float(pca.explained_variance_[index]), float(pca.explained_variance_ratio_[index])),
    )
    print("Writing ECA assets…", flush=True)
    eca_record = build_method_assets(
        args.output, "eca", "SpectreFaces", mean, components_eca, weights_eca, cumulative_eca,
        lambda index: (avg_eigenvalue, avg_explained_variance),
    )

    # Both methods converge to the same final image at k = MAX_COMPONENTS
    # (H is orthogonal, nothing is truncated) -- one shared static preview
    # for the pre-JS fallback, order of summation doesn't matter.
    final = mean + components_pca[:MAX_COMPONENTS].T @ weights_pca[:MAX_COMPONENTS]
    save_gray(args.output / "mean.png", mean.reshape(SIZE, SIZE))
    save_gray(args.output / "reconstruction.png", final.reshape(SIZE, SIZE))

    provenance = [
        {"id": index, "subject": subj, "code": code}
        for index, (subj, code, _) in enumerate(training_items)
    ]
    (args.cache / "provenance.json").write_text(json.dumps(provenance, indent=2))

    manifest = {
        "schemaVersion": 2,
        "dataset": "Extended Yale Face Database B (Cropped)",
        "datasetUrl": "http://vision.ucsd.edu/~leekc/ExtYaleDatabase/ExtYaleB.html",
        "seed": SEED,
        "sampleCount": len(training_items),
        "width": SIZE,
        "height": SIZE,
        "flatteningOrder": "row-major",
        "grayscale": "8-bit grayscale (dataset is monochrome)",
        "defaultDimensions": DEFAULT_COMPONENTS,
        "maxDimensions": MAX_COMPONENTS,
        "exposedComponentCount": EXPOSED_COMPONENTS,
        "mean": "/eigenfaces/mean.png",
        "reconstruction": "/eigenfaces/reconstruction.png",
        "methods": {
            "pca": pca_record,
            "eca": eca_record,
        },
        "portraitAlignment": {
            "method": "Pre-cropped/aligned by the Extended Yale B curators; "
                      "no additional alignment performed",
            "heldOutImage": f"{portrait_subj}-{portrait_code}",
        },
    }
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(
        f"Wrote assets with default={DEFAULT_COMPONENTS}, max={MAX_COMPONENTS}; "
        f"PCA variance@{MAX_COMPONENTS}={cumulative_pca[-1]:.4f}, "
        f"ECA variance@{MAX_COMPONENTS}={cumulative_eca[-1]:.4f}",
        flush=True,
    )


if __name__ == "__main__":
    main()
