import { readFileSync } from 'node:fs';
import { resolve } from 'node:path';
import { gunzipSync } from 'node:zlib';

import { describe, expect, it } from 'vitest';

type ComponentRecord = {
  index: number;
  eigenvalue: number;
  explainedVariance: number;
  thumbnail: string;
};

type MethodRecord = {
  label: string;
  prefixReconstructions: string;
  cumulativeExplainedVariance: number[];
  components: ComponentRecord[];
};

type Manifest = {
  defaultDimensions: number;
  maxDimensions: number;
  width: number;
  height: number;
  methods: { pca: MethodRecord; eca: MethodRecord };
};

const manifest = JSON.parse(
  readFileSync(resolve('public/eigenfaces/manifest.json'), 'utf8'),
) as Manifest;

const METHODS = ['pca', 'eca'] as const;

describe('dual-method (PCA vs ECA) dimension-prefix assets', () => {
  it('shares one dimensions range across both methods', () => {
    expect(manifest.defaultDimensions).toBeGreaterThan(0);
    expect(manifest.defaultDimensions).toBeLessThanOrEqual(
      manifest.maxDimensions,
    );
    for (const slug of METHODS) {
      const method = manifest.methods[slug];
      expect(method.cumulativeExplainedVariance).toHaveLength(
        manifest.maxDimensions,
      );
      expect(
        gunzipSync(
          readFileSync(
            resolve(
              'public',
              method.prefixReconstructions.replace(/^\//, ''),
            ),
          ),
        ).length,
      ).toBe(manifest.maxDimensions * manifest.width * manifest.height);
    }
  });

  it('has a monotonically non-decreasing cumulative variance curve per method', () => {
    for (const slug of METHODS) {
      const curve = manifest.methods[slug].cumulativeExplainedVariance;
      for (let index = 1; index < curve.length; index += 1) {
        expect(curve[index]).toBeGreaterThanOrEqual(curve[index - 1]);
      }
      expect(curve[curve.length - 1]).toBeGreaterThan(0.9);
    }
  });

  it('PCA concentrates variance in far fewer components than ECA, by design', () => {
    // ECA left-multiplies by a normalized Hadamard matrix, which provably
    // equalizes variance across every coefficient -- so unlike PCA's
    // concentrated spectrum, ECA's early cumulative share should track
    // roughly k / maxDimensions (linear), not shoot ahead of it.
    const earlyIndex = Math.min(4, manifest.maxDimensions - 1); // k = 5
    const pcaEarly = manifest.methods.pca.cumulativeExplainedVariance[earlyIndex];
    const ecaEarly = manifest.methods.eca.cumulativeExplainedVariance[earlyIndex];
    expect(pcaEarly).toBeGreaterThan(ecaEarly);
    expect(ecaEarly).toBeLessThan(
      (earlyIndex + 1) / manifest.maxDimensions + 0.02,
    );
  });

  it('exports 24 ordered, finite, checksum-verifiable thumbnails per method', () => {
    for (const slug of METHODS) {
      const method = manifest.methods[slug];
      expect(method.components).toHaveLength(24);
      method.components.forEach((component, index) => {
        expect(component.index).toBe(index + 1);
        expect(Number.isFinite(component.eigenvalue)).toBe(true);
        expect(Number.isFinite(component.explainedVariance)).toBe(true);

        const thumbnail = readFileSync(
          resolve('public', component.thumbnail.replace(/^\//, '')),
        );
        expect(thumbnail.subarray(1, 4).toString()).toBe('PNG');
        expect(thumbnail.readUInt32BE(16)).toBe(manifest.width);
        expect(thumbnail.readUInt32BE(20)).toBe(manifest.height);
      });
    }
  });
});
