/** Micro-profile of one worker's gradient step and one env step. npx tsx training/profile.ts */
import RAPIER from '@dimforge/rapier3d-compat'
import { ACTION_SIZE, ByteEnv, OBS_SIZE } from '../src/sim/env'
import { MLP } from '../src/sim/policy'
import { createRng } from '../src/sim/random'
import { backwardBatch, forwardBatch } from './nn'

await RAPIER.init()
const rng = createRng(1)
for (const hidden of [[256, 128], [128, 128], [128, 64]]) {
  const net = new MLP([OBS_SIZE, ...hidden, ACTION_SIZE]).init(rng.normal)
  const n = 96
  const x = Float32Array.from({ length: n * OBS_SIZE }, rng.normal)
  const g = new Float32Array(net.size)
  const t0 = performance.now()
  for (let r = 0; r < 20; r++) {
    const c = forwardBatch(net, x, n)
    backwardBatch(net, c, c.output, g)
  }
  console.log(`hidden ${hidden}: fwd+bwd ${((performance.now() - t0) / 20 / n * 1000).toFixed(1)} µs/sample (${net.size} params)`)
}
const env = new ByteEnv(RAPIER, 3)
const obs = new Float32Array(OBS_SIZE)
env.reset(obs)
const a = new Float32Array(ACTION_SIZE)
const t1 = performance.now()
let steps = 0
for (; steps < 2000; steps++) {
  for (let i = 0; i < ACTION_SIZE; i++) a[i] = rng.normal() * 0.3
  const r = env.step(a, obs)
  if (r.terminated || r.truncated) env.reset(obs)
}
console.log(`env step ${((performance.now() - t1) / steps * 1000).toFixed(0)} µs (3 physics steps + obs + rewards)`)
