import type { MLP } from '../src/sim/policy'

/** Activations kept from a batched forward pass, needed for backprop. */
export interface ForwardCache {
  n: number
  /** Input to each layer (layer 0's input is the batch itself). */
  inputs: Float32Array[]
  /** Pre-activation outputs of each layer. */
  pre: Float32Array[]
  output: Float32Array
}

/** Batched forward pass over `n` row-major samples. */
export function forwardBatch(net: MLP, x: Float32Array, n: number): ForwardCache {
  const inputs: Float32Array[] = []
  const pre: Float32Array[] = []
  let a = x
  const p = net.params
  net.layout.forEach(({ w, b, inSize, outSize }, l) => {
    const last = l === net.layout.length - 1
    const z = new Float32Array(n * outSize)
    for (let s = 0; s < n; s++) {
      for (let o = 0; o < outSize; o++) z[s * outSize + o] = p[b + o] + dot(p, w + o * inSize, a, s * inSize, inSize)
    }
    inputs.push(a)
    pre.push(z)
    if (!last) {
      const act = new Float32Array(z.length)
      for (let i = 0; i < z.length; i++) act[i] = z[i] > 0 ? z[i] : Math.expm1(z[i])
      a = act
    } else {
      a = z
    }
  })
  return { n, inputs, pre, output: a }
}

/** Contiguous dot product, 4-way unrolled. */
function dot(x: Float32Array, xo: number, y: Float32Array, yo: number, len: number): number {
  let a0 = 0
  let a1 = 0
  let a2 = 0
  let a3 = 0
  let i = 0
  const tail = len & ~3
  for (; i < tail; i += 4) {
    a0 += x[xo + i] * y[yo + i]
    a1 += x[xo + i + 1] * y[yo + i + 1]
    a2 += x[xo + i + 2] * y[yo + i + 2]
    a3 += x[xo + i + 3] * y[yo + i + 3]
  }
  for (; i < len; i++) a0 += x[xo + i] * y[yo + i]
  return a0 + a1 + a2 + a3
}

/** rows × cols → cols × rows. */
function transpose(m: Float32Array, mo: number, rows: number, cols: number): Float32Array {
  const t = new Float32Array(rows * cols)
  for (let r = 0; r < rows; r++) for (let c = 0; c < cols; c++) t[c * rows + r] = m[mo + r * cols + c]
  return t
}

/**
 * Accumulate dLoss/dParams into `grad` given dLoss/dOutput. Every product is
 * arranged as a contiguous dot product (after transposing the activations
 * and weights once), so each gradient entry is written exactly once.
 */
export function backwardBatch(net: MLP, cache: ForwardCache, dOut: Float32Array, grad: Float32Array) {
  const { n } = cache
  const p = net.params
  let dz = dOut
  for (let l = net.layout.length - 1; l >= 0; l--) {
    const { w, b, inSize, outSize } = net.layout[l]
    const aT = transpose(cache.inputs[l], 0, n, inSize) // inSize × n
    const dzT = transpose(dz, 0, n, outSize) // outSize × n
    for (let o = 0; o < outSize; o++) {
      let gb = 0
      for (let s = 0; s < n; s++) gb += dzT[o * n + s]
      grad[b + o] += gb
      const row = w + o * inSize
      for (let i = 0; i < inSize; i++) grad[row + i] += dot(dzT, o * n, aT, i * n, n)
    }
    if (l === 0) break
    // dA = dZ · W, using Wᵀ so each entry is a contiguous dot product.
    const wT = transpose(p, w, outSize, inSize) // inSize × outSize
    const da = new Float32Array(n * inSize)
    const prevPre = cache.pre[l - 1]
    for (let s = 0; s < n; s++) {
      for (let i = 0; i < inSize; i++) {
        const k = s * inSize + i
        const z = prevPre[k]
        const g = dot(dz, s * outSize, wT, i * outSize, outSize)
        da[k] = z > 0 ? g : g * Math.exp(z) // through the previous layer's ELU
      }
    }
    dz = da
  }
}

export class Adam {
  private readonly m: Float32Array
  private readonly v: Float32Array
  private t = 0

  constructor(
    size: number,
    public lr = 3e-4,
    private readonly beta1 = 0.9,
    private readonly beta2 = 0.999,
    private readonly eps = 1e-8,
  ) {
    this.m = new Float32Array(size)
    this.v = new Float32Array(size)
  }

  step(params: Float32Array, grad: Float32Array) {
    this.t++
    const c1 = 1 - this.beta1 ** this.t
    const c2 = 1 - this.beta2 ** this.t
    for (let i = 0; i < params.length; i++) {
      const g = grad[i]
      this.m[i] = this.beta1 * this.m[i] + (1 - this.beta1) * g
      this.v[i] = this.beta2 * this.v[i] + (1 - this.beta2) * g * g
      params[i] -= (this.lr * (this.m[i] / c1)) / (Math.sqrt(this.v[i] / c2) + this.eps)
    }
  }
}

/** Scale `grads` in place so their combined L2 norm is at most `maxNorm`. */
export function clipGradNorm(grads: Float32Array[], maxNorm: number): number {
  let sq = 0
  for (const g of grads) for (let i = 0; i < g.length; i++) sq += g[i] * g[i]
  const norm = Math.sqrt(sq)
  if (norm > maxNorm) {
    const k = maxNorm / (norm + 1e-6)
    for (const g of grads) for (let i = 0; i < g.length; i++) g[i] *= k
  }
  return norm
}
