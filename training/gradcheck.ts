/** Finite-difference check of backwardBatch. npx tsx training/gradcheck.ts */
import { MLP } from '../src/sim/policy'
import { backwardBatch, forwardBatch } from './nn'
import { createRng } from '../src/sim/random'

const rng = createRng(3)
const net = new MLP([5, 7, 6, 3]).init(rng.normal, 1)
const n = 4
const x = Float32Array.from({ length: n * 5 }, rng.normal)
const target = Float32Array.from({ length: n * 3 }, rng.normal)

// Loss = 0.5 Σ (y − target)²
const loss = () => {
  const y = forwardBatch(net, x, n).output
  let l = 0
  for (let i = 0; i < y.length; i++) l += 0.5 * (y[i] - target[i]) ** 2
  return l
}

const cache = forwardBatch(net, x, n)
const dOut = cache.output.map((y, i) => y - target[i])
const grad = new Float32Array(net.size)
backwardBatch(net, cache, dOut, grad)

let worst = 0
for (let i = 0; i < net.size; i++) {
  const keep = net.params[i]
  const h = 1e-2
  net.params[i] = keep + h
  const up = loss()
  net.params[i] = keep - h
  const down = loss()
  net.params[i] = keep
  const numeric = (up - down) / (2 * h)
  const rel = Math.abs(numeric - grad[i]) / Math.max(1e-3, Math.abs(numeric) + Math.abs(grad[i]))
  worst = Math.max(worst, rel)
}
console.log(`max relative gradient error: ${worst.toExponential(2)} ${worst < 1e-2 ? '✓' : '✗'}`)
