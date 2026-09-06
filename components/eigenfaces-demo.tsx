'use client';

import { useEffect, useRef, useState } from 'react';
import Image from 'next/image';

import {
  Carousel,
  CarouselContent,
  CarouselItem,
  type CarouselApi,
} from '@/components/ui/carousel';
import { Slider } from '@/components/ui/slider';
import { carouselPresentation, componentIsWithinPrefix } from '@/lib/carousel';

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
  dataset: string;
  datasetUrl: string;
  sampleCount: number;
  width: number;
  height: number;
  defaultDimensions: number;
  maxDimensions: number;
  mean: string;
  reconstruction: string;
  methods: { pca: MethodRecord; eca: MethodRecord };
};

type LoadedModel = {
  manifest: Manifest;
  prefixByMethod: { pca: Uint8Array; eca: Uint8Array };
};

function publicAssetUrl(path: string) {
  if (typeof document === 'undefined') return path;
  return new URL(path.replace(/^\//, ''), document.baseURI).pathname;
}

function drawGrayscaleBytes(
  canvas: HTMLCanvasElement,
  bytes: Uint8Array,
  offset: number,
  width: number,
  height: number,
) {
  const context = canvas.getContext('2d');
  if (!context) return;
  const image = context.createImageData(width, height);
  const pixelCount = width * height;
  for (let index = 0; index < pixelCount; index += 1) {
    const value = bytes[offset + index];
    const pixelOffset = index * 4;
    image.data[pixelOffset] = value;
    image.data[pixelOffset + 1] = value;
    image.data[pixelOffset + 2] = value;
    image.data[pixelOffset + 3] = 255;
  }
  context.putImageData(image, 0, 0);
}

async function fetchUint8(url: string) {
  const response = await fetch(url);
  if (!response.ok) throw new Error(`Could not load ${url}`);
  const decompressed = response.body?.pipeThrough(
    new DecompressionStream('gzip'),
  );
  if (!decompressed) throw new Error(`Could not decompress ${url}`);
  return new Uint8Array(await new Response(decompressed).arrayBuffer());
}

function decodePrefixDeltas(payload: Uint8Array, pixelCount: number) {
  for (let offset = pixelCount; offset < payload.length; offset += 1) {
    payload[offset] = (payload[offset] + payload[offset - pixelCount]) & 0xff;
  }
  return payload;
}

function applyCarouselDepth(api: CarouselApi) {
  if (!api) return;
  const slideNodes = api.slideNodes();
  const root = api.rootNode().getBoundingClientRect();
  const center = root.left + root.width / 2;
  const step = slideNodes[0]?.offsetWidth || 1;
  const distances = slideNodes.map((node) => {
    const bounds = node.getBoundingClientRect();
    return Math.abs(bounds.left + bounds.width / 2 - center) / step;
  });
  const nearestDistance = Math.min(...distances);

  slideNodes.forEach((node, index) => {
    const { scale, opacity, brightness } = carouselPresentation(
      Math.max(0, distances[index] - nearestDistance),
    );
    node.style.setProperty('--slide-scale', scale.toFixed(3));
    node.style.setProperty('--slide-opacity', opacity.toFixed(3));
    node.style.setProperty('--slide-brightness', brightness.toFixed(3));
    node.style.zIndex = String(Math.round(scale * 100));
  });
}

export function EigenfacesDemo() {
  const [model, setModel] = useState<LoadedModel | null>(null);
  const [dimensions, setDimensions] = useState(2048);
  const [carouselApi, setCarouselApi] = useState<CarouselApi>();
  const [selectedSlide, setSelectedSlide] = useState(0);
  const [error, setError] = useState<string | null>(null);
  const pcaCanvasRef = useRef<HTMLCanvasElement>(null);
  const ecaCanvasRef = useRef<HTMLCanvasElement>(null);

  useEffect(() => {
    let cancelled = false;
    async function loadModel() {
      try {
        const manifestResponse = await fetch(
          publicAssetUrl('/eigenfaces/manifest.json'),
        );
        if (!manifestResponse.ok)
          throw new Error('The eigenspace is unavailable.');
        const manifest = (await manifestResponse.json()) as Manifest;
        const [pcaPrefix, ecaPrefix] = await Promise.all([
          fetchUint8(publicAssetUrl(manifest.methods.pca.prefixReconstructions)),
          fetchUint8(publicAssetUrl(manifest.methods.eca.prefixReconstructions)),
        ]);
        const expectedLength =
          manifest.width * manifest.height * manifest.maxDimensions;
        if (
          pcaPrefix.length !== expectedLength ||
          ecaPrefix.length !== expectedLength
        ) {
          throw new Error('The eigenspace data has an unexpected size.');
        }
        const pixelCount = manifest.width * manifest.height;
        decodePrefixDeltas(pcaPrefix, pixelCount);
        decodePrefixDeltas(ecaPrefix, pixelCount);
        if (!cancelled) {
          setModel({ manifest, prefixByMethod: { pca: pcaPrefix, eca: ecaPrefix } });
          setDimensions(manifest.defaultDimensions);
        }
      } catch (reason) {
        if (!cancelled)
          setError(
            reason instanceof Error
              ? reason.message
              : 'The eigenspace is unavailable.',
          );
      }
    }
    void loadModel();
    return () => {
      cancelled = true;
    };
  }, []);

  useEffect(() => {
    if (!carouselApi) return;
    const update = () => applyCarouselDepth(carouselApi);
    const select = () => setSelectedSlide(carouselApi.selectedScrollSnap());
    update();
    select();
    carouselApi.on('scroll', update);
    carouselApi.on('reInit', update);
    carouselApi.on('select', select);
    return () => {
      carouselApi.off('scroll', update);
      carouselApi.off('reInit', update);
      carouselApi.off('select', select);
    };
  }, [carouselApi]);

  useEffect(() => {
    if (!model) return;
    const { width, height } = model.manifest;
    const pixelCount = width * height;
    const offset = (dimensions - 1) * pixelCount;
    const frame = requestAnimationFrame(() => {
      if (pcaCanvasRef.current)
        drawGrayscaleBytes(
          pcaCanvasRef.current,
          model.prefixByMethod.pca,
          offset,
          width,
          height,
        );
      if (ecaCanvasRef.current)
        drawGrayscaleBytes(
          ecaCanvasRef.current,
          model.prefixByMethod.eca,
          offset,
          width,
          height,
        );
    });
    return () => cancelAnimationFrame(frame);
  }, [dimensions, model]);

  const pcaVariance =
    model?.manifest.methods.pca.cumulativeExplainedVariance[dimensions - 1] ??
    null;
  const ecaVariance =
    model?.manifest.methods.eca.cumulativeExplainedVariance[dimensions - 1] ??
    null;
  const pcaComponents = model?.manifest.methods.pca.components ?? [];
  const ecaComponents = model?.manifest.methods.eca.components ?? [];

  return (
    <main className="eigenfaces-page">
      <div className="ambient-glow" aria-hidden="true" />
      <header className="hero-block">
        <h1>
          Eigenfaces <span className="hero-vs">vs</span> SpectreFaces
        </h1>
        <p>Face It: It’s Just Linear Algebra</p>
      </header>

      <section
        className="reconstruction-area"
        aria-label="Side-by-side Eigenfaces vs SpectreFaces reconstruction"
      >
        <div className="reconstruction-pair">
          <figure className="reconstruction-figure">
            <figcaption className="figure-topline">
              <span className="eyebrow">Eigenfaces (PCA)</span>
            </figcaption>
            <div className="reconstruction-stage">
              {!model && !error && (
                <Image
                  className="reconstruction-fallback"
                  src={publicAssetUrl('/eigenfaces/reconstruction.png')}
                  alt="Grayscale preview of the reconstructed input portrait"
                  width={128}
                  height={128}
                  priority
                  unoptimized
                />
              )}
              <canvas
                ref={pcaCanvasRef}
                className={
                  model
                    ? 'reconstruction-canvas is-ready'
                    : 'reconstruction-canvas'
                }
                width={model?.manifest.width ?? 128}
                height={model?.manifest.height ?? 128}
                role="img"
                aria-label="PCA reconstruction of the held-out portrait"
              />
              <div className="scan-line" aria-hidden="true" />
            </div>
            <div className="reconstruction-footer">
              <div className="variance-metric">
                <span className="footer-kicker">Variance retained</span>
                <strong>
                  {pcaVariance !== null ? `${(pcaVariance * 100).toFixed(1)}%` : '—'}
                </strong>
              </div>
            </div>
          </figure>

          <figure className="reconstruction-figure">
            <figcaption className="figure-topline">
              <span className="eyebrow">SpectreFaces (ECA)</span>
            </figcaption>
            <div className="reconstruction-stage">
              {!model && !error && (
                <Image
                  className="reconstruction-fallback"
                  src={publicAssetUrl('/eigenfaces/reconstruction.png')}
                  alt="Grayscale preview of the reconstructed input portrait"
                  width={128}
                  height={128}
                  unoptimized
                />
              )}
              <canvas
                ref={ecaCanvasRef}
                className={
                  model
                    ? 'reconstruction-canvas is-ready'
                    : 'reconstruction-canvas'
                }
                width={model?.manifest.width ?? 128}
                height={model?.manifest.height ?? 128}
                role="img"
                aria-label="ECA reconstruction of the held-out portrait"
              />
              <div className="scan-line" aria-hidden="true" />
            </div>
            <div className="reconstruction-footer">
              <div className="variance-metric">
                <span className="footer-kicker">Variance retained</span>
                <strong>
                  {ecaVariance !== null ? `${(ecaVariance * 100).toFixed(1)}%` : '—'}
                </strong>
              </div>
            </div>
          </figure>
        </div>

        {error && (
          <div className="model-error" role="alert">
            <span>Eigenspace unavailable</span>
            <small>{error}</small>
          </div>
        )}

        <div className="dimensions-bar">
          <span className="footer-kicker">
            Components used <output>{dimensions}</output>
          </span>
          <div className="dimensions-slider-shell">
            <span>1</span>
            <Slider
              aria-label="Number of principal components used in both reconstructions"
              min={1}
              max={model?.manifest.maxDimensions ?? 2048}
              step={1}
              value={[dimensions]}
              disabled={!model}
              onValueChange={(next) =>
                setDimensions(Math.round(Array.isArray(next) ? next[0] : next))
              }
            />
            <span>{model?.manifest.maxDimensions ?? 2048}</span>
          </div>
        </div>
      </section>

      <section className="basis-section" aria-labelledby="basis-title">
        <div className="basis-heading">
          <h2 id="basis-title">Principal components, paired</h2>
        </div>

        <Carousel
          className="component-carousel"
          setApi={setCarouselApi}
          opts={{
            align: 'center',
            loop: true,
            startIndex: 0,
            duration: 40,
            skipSnaps: true,
            slidesToScroll: 1,
          }}
          aria-label="Mean face and paired principal components"
        >
          <CarouselContent className="carousel-track">
            <CarouselItem
              className={`component-slide${selectedSlide === 0 ? ' is-selected' : ''}`}
            >
              <figure className="component-tile mean-tile">
                <Image
                  src={publicAssetUrl('/eigenfaces/mean.png')}
                  alt="Average face across the Extended Yale B training sample"
                  width={128}
                  height={128}
                  unoptimized
                />
                <figcaption>
                  <span>Mean</span>
                </figcaption>
                <button
                  type="button"
                  className="tile-activator"
                  aria-label="Center the mean face"
                  onClick={() => carouselApi?.scrollTo(0)}
                />
              </figure>
            </CarouselItem>

            {pcaComponents.map((pcaComponent, index) => {
              const ecaComponent = ecaComponents[index];
              const slideIndex = index + 1;
              const selected = selectedSlide === slideIndex;
              const excluded = !componentIsWithinPrefix(
                pcaComponent.index,
                dimensions,
              );
              return (
                <CarouselItem
                  className={`component-slide${selected ? ' is-selected' : ''}`}
                  key={pcaComponent.index}
                >
                  <figure
                    className={`component-tile pair-tile${excluded ? ' is-excluded' : ''}`}
                  >
                    <div className="pair-images">
                      <div className="pair-half">
                        <Image
                          src={publicAssetUrl(pcaComponent.thumbnail)}
                          alt={`Eigenfaces component ${pcaComponent.index}`}
                          width={128}
                          height={128}
                          unoptimized
                        />
                        <span className="pair-label">Eigenfaces</span>
                      </div>
                      <div className="pair-half">
                        <Image
                          src={publicAssetUrl(ecaComponent.thumbnail)}
                          alt={`SpectreFaces component ${ecaComponent.index}`}
                          width={128}
                          height={128}
                          unoptimized
                        />
                        <span className="pair-label">SpectreFaces</span>
                      </div>
                    </div>
                    <figcaption>
                      <span>Component {String(pcaComponent.index).padStart(2, '0')}</span>
                      <small>
                        {(pcaComponent.explainedVariance * 100).toFixed(1)}% /{' '}
                        {(ecaComponent.explainedVariance * 100).toFixed(2)}%
                      </small>
                    </figcaption>
                    <button
                      type="button"
                      className="tile-activator"
                      aria-label={`Center component ${pcaComponent.index}`}
                      onClick={() => carouselApi?.scrollTo(slideIndex)}
                    />
                  </figure>
                </CarouselItem>
              );
            })}

            {!model &&
              Array.from({ length: 24 }, (_, index) => (
                <CarouselItem
                  className="component-slide"
                  key={index}
                  aria-hidden="true"
                >
                  <div className="component-tile loading-tile">
                    <div className="loading-face" />
                    <span>Component {String(index + 1).padStart(2, '0')}</span>
                  </div>
                </CarouselItem>
              ))}
          </CarouselContent>
        </Carousel>
      </section>
    </main>
  );
}
