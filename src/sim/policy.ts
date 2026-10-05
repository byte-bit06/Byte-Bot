/**
 * A tiny dependency-free MLP (ELU hidden layers, linear output) plus the
 * observation normaliser, shared by the trainer and the website.
 *
 * Parameter layout, per layer: weights (out × in, row-major) then biases.
 */

export const elu = (x: number) => (x > 0 ? x : Math.expm1(x))

export class MLP {
  readonly sizes: number[]
  readonly params: Float32Array
  /** Offsets of each layer's weights and biases inside `params`. */
  readonly layout: { w: number; b: number; inSize: number; outSize: number }[] = []

  constructor(sizes: number[], params?: Float32Array) {
    this.sizes = sizes
    let n = 0
    for (let l = 0; l < sizes.length - 1; l++) {
      const inSize = sizes[l]
      const outSize = sizes[l + 1]
      this.layout.push({ w: n, b: n + inSize * outSize, inSize, outSize })
      n += inSize * outSize + outSize
    }
    this.params = params ?? new Float32Array(n)
    if (this.params.length !== n) throw new Error(`MLP expects ${n} params, got ${this.params.length}`)
  }

  get size() {
    return this.params.length
  }

  /** Orthogonal-ish init: scaled Gaussian weights, zero biases. */
  init(rand: () => number, outputGain = 0.01) {
    this.layout.forEach(({ w, inSize, outSize }, l) => {
      const last = l === this.layout.length - 1
      const std = (last ? outputGain : Math.SQRT2) / Math.sqrt(inSize)
      for (let i = 0; i < inSize * outSize; i++) this.params[w + i] = rand() * std
    })
    return this
  }

  /** Single-sample forward pass. `scratch` avoids allocations in hot loops. */
  forward(input: Float32Array, out: Float32Array, scratch?: Float32Array[]): Float32Array {
    let x = input
    this.layout.forEach(({ w, b, inSize, outSize }, l) => {
      const last = l === this.layout.length - 1
      const y = last ? out : (scratch?.[l] ?? new Float32Array(outSize))
      const p = this.params
      for (let o = 0; o < outSize; o++) {
        let acc = p[b + o]
        const row = w + o * inSize
        for (let i = 0; i < inSize; i++) acc += p[row + i] * x[i]
        y[o] = last ? acc : elu(acc)
      }
      x = y
    })
    return out
  }

  /** Reusable hidden-layer buffers for `forward`. */
  scratch(): Float32Array[] {
    return this.layout.slice(0, -1).map(({ outSize }) => new Float32Array(outSize))
  }
}

/** Running mean / variance of observations (Welford, mergeable across workers). */
export class ObsNormalizer {
  mean: Float64Array
  m2: Float64Array
  count = 1e-4

  constructor(size: number) {
    this.mean = new Float64Array(size)
    this.m2 = new Float64Array(size).fill(1e-4)
  }

  /** Merge a batch's (count, mean, M2) statistics. */
  merge(count: number, mean: ArrayLike<number>, m2: ArrayLike<number>) {
    const total = this.count + count
    for (let i = 0; i < this.mean.length; i++) {
      const delta = mean[i] - this.mean[i]
      this.mean[i] += (delta * count) / total
      this.m2[i] += m2[i] + (delta * delta * this.count * count) / total
    }
    this.count = total
  }

  std(): Float32Array {
    const s = new Float32Array(this.mean.length)
    for (let i = 0; i < s.length; i++) s[i] = Math.sqrt(this.m2[i] / this.count + 1e-8)
    return s
  }
}

export function normalizeInto(obs: Float32Array, mean: ArrayLike<number>, std: ArrayLike<number>, out: Float32Array) {
  for (let i = 0; i < obs.length; i++) out[i] = Math.max(-5, Math.min(5, (obs[i] - mean[i]) / std[i]))
  return out
}

// ---------------------------------------------------------------------------
// Serialised policy (what the trainer exports and the website loads)
// ---------------------------------------------------------------------------

export interface PolicyMeta {
  iteration: number
  envSteps: number
  wallSeconds: number
  meanReturn: number
  meanEpisodeSeconds: number
  goalsPerEpisode: number
  fallRate: number
  terrainLevel: number
}

export interface PolicyFile {
  version: 1
  sizes: number[]
  /** Base64-encoded little-endian Float32 parameters. */
  params: string
  obsMean: number[]
  obsStd: number[]
  logStd: number[]
  meta: PolicyMeta
}

export function encodeFloat32(a: Float32Array): string {
  const bytes = new Uint8Array(a.buffer, a.byteOffset, a.byteLength)
  let s = ''
  for (let i = 0; i < bytes.length; i += 0x8000) s += String.fromCharCode(...bytes.subarray(i, i + 0x8000))
  return btoa(s)
}

export function decodeFloat32(b64: string): Float32Array {
  const s = atob(b64)
  const bytes = new Uint8Array(s.length)
  for (let i = 0; i < s.length; i++) bytes[i] = s.charCodeAt(i)
  return new Float32Array(bytes.buffer)
}

/** Deterministic (mean-action) controller for deployment on the website. */
export class Policy {
  readonly net: MLP
  readonly meta: PolicyMeta
  private readonly mean: Float32Array
  private readonly std: Float32Array
  private readonly normalized: Float32Array
  private readonly scratch: Float32Array[]

  constructor(file: PolicyFile) {
    this.net = new MLP(file.sizes, decodeFloat32(file.params))
    this.meta = file.meta
    this.mean = Float32Array.from(file.obsMean)
    this.std = Float32Array.from(file.obsStd)
    this.normalized = new Float32Array(file.sizes[0])
    this.scratch = this.net.scratch()
  }

  act(obs: Float32Array, out: Float32Array): Float32Array {
    normalizeInto(obs, this.mean, this.std, this.normalized)
    return this.net.forward(this.normalized, out, this.scratch)
  }
}
