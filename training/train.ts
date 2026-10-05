/**
 * PPO trainer for Byte.
 *
 *   npx tsx training/train.ts --run walk-v1 [--workers 22] [--envs 12] [--iterations 3000] [--resume]
 *
 * Rollouts and gradients are computed in parallel worker threads (one Rapier
 * world per env, same physics as the website). This thread owns the networks,
 * the optimiser, the curriculum, logging and checkpoints.
 */
import { Worker } from 'node:worker_threads'
import { cpus } from 'node:os'
import { mkdirSync, readFileSync, writeFileSync, appendFileSync, existsSync } from 'node:fs'
import { join } from 'node:path'
import { parseArgs } from 'node:util'
import { ACTION_SIZE, CONTROL_DT, OBS_SIZE, REWARD_TERMS, type Curriculum } from '../src/sim/env'
import { MLP, ObsNormalizer, encodeFloat32, decodeFloat32, type PolicyFile } from '../src/sim/policy'
import { createRng } from '../src/sim/random'
import { Adam, clipGradNorm } from './nn'
import { PPO, type EpisodeStats, type FromWorker, type NetParams, type ToWorker } from './protocol'

const { values: args } = parseArgs({
  options: {
    run: { type: 'string', default: 'walk-v1' },
    workers: { type: 'string', default: String(Math.max(1, cpus().length - 2)) },
    envs: { type: 'string', default: '12' },
    iterations: { type: 'string', default: '3000' },
    resume: { type: 'boolean', default: false },
    seed: { type: 'string', default: '1' },
  },
})

const RUN_DIR = join('training', 'runs', args.run!)
mkdirSync(RUN_DIR, { recursive: true })
const numWorkers = Number(args.workers)
const envsPerWorker = Number(args.envs)
const maxIterations = Number(args.iterations)

// ---------------------------------------------------------------------------
// Curriculum: harder goals, then domain randomisation, then rough terrain
// ---------------------------------------------------------------------------

// Terrain arrives early: each env samples a terrain level up to the current
// maximum, so harder ground is added without forgetting flat walking.
const LADDER: Curriculum[] = [
  { terrainLevel: 0, goalDistance: [0.8, 1.5], goalAngle: 0.4, randomization: 0 },
  { terrainLevel: 0, goalDistance: [1, 2.5], goalAngle: 0.9, randomization: 0 },
  { terrainLevel: 0, goalDistance: [1, 3], goalAngle: 1.8, randomization: 0.25 },
  { terrainLevel: 1, goalDistance: [1, 3], goalAngle: 1.8, randomization: 0.3 },
  { terrainLevel: 2, goalDistance: [1, 3.5], goalAngle: Math.PI, randomization: 0.4 },
  { terrainLevel: 3, goalDistance: [1, 3.5], goalAngle: Math.PI, randomization: 0.5 },
  { terrainLevel: 3, goalDistance: [1, 4], goalAngle: Math.PI, randomization: 0.8 },
]
// Measured per minute of simulated experience over the last few iterations,
// counted from every rollout step. (Per-episode rates are biased: after a
// restart, episodes finish in synchronised waves.) Training actions are
// stochastic, so these thresholds are looser than deterministic evaluation.
const PROMOTE = { goalsPerMinute: 8, maxFallsPerMinute: 1.5, cooldownIterations: 40 }
const DEMOTE = { fallsPerMinute: 4 }
const RATE_WINDOW = 20

// ---------------------------------------------------------------------------
// State
// ---------------------------------------------------------------------------

interface TrainState {
  iteration: number
  envSteps: number
  wallSeconds: number
  level: number
  lastPromotion: number
  lr: number
  policy: string
  value: string
  logStd: number[]
  obs: { count: number; mean: number[]; m2: number[] }
}

const rng = createRng(Number(args.seed))
const policy = new MLP([OBS_SIZE, ...PPO.hidden, ACTION_SIZE]).init(rng.normal, 0.01)
const value = new MLP([OBS_SIZE, ...PPO.hidden, 1]).init(rng.normal, 1)
const logStd = new Float32Array(ACTION_SIZE).fill(PPO.initLogStd)
const normalizer = new ObsNormalizer(OBS_SIZE)
let iteration = 0
let envSteps = 0
let wallSeconds = 0
let level = 0
let lastPromotion = 0
let lr: number = PPO.learningRate

const statePath = join(RUN_DIR, 'state.json')
if (args.resume && existsSync(statePath)) {
  const s = JSON.parse(readFileSync(statePath, 'utf8')) as TrainState
  policy.params.set(decodeFloat32(s.policy))
  value.params.set(decodeFloat32(s.value))
  logStd.set(s.logStd)
  normalizer.count = s.obs.count
  normalizer.mean.set(s.obs.mean)
  normalizer.m2.set(s.obs.m2)
  ;({ iteration, envSteps, wallSeconds, level, lastPromotion, lr } = s)
  console.log(`resumed ${args.run} at iteration ${iteration}, level ${level}`)
}

const optPolicy = new Adam(policy.size, lr)
const optValue = new Adam(value.size, lr)
const optLogStd = new Adam(ACTION_SIZE, lr)

// ---------------------------------------------------------------------------
// Workers
// ---------------------------------------------------------------------------

const workers = Array.from(
  { length: numWorkers },
  (_, i) =>
    new Worker(new URL('./worker.ts', import.meta.url), {
      workerData: { numEnvs: envsPerWorker, seed: Number(args.seed) * 7919 + i + iteration * 104729 },
      execArgv: ['--import', 'tsx'],
    }),
)

function request<T extends FromWorker['type']>(w: Worker, msg: ToWorker | null, type: T) {
  return new Promise<Extract<FromWorker, { type: T }>>((resolve, reject) => {
    const onMessage = (m: FromWorker) => {
      if (m.type !== type) return
      w.off('message', onMessage)
      w.off('error', reject)
      resolve(m as Extract<FromWorker, { type: T }>)
    }
    w.on('message', onMessage)
    w.once('error', reject)
    if (msg) w.postMessage(msg)
  })
}
const all = <T extends FromWorker['type']>(msg: ToWorker | null, type: T) =>
  Promise.all(workers.map((w) => request(w, msg, type)))

const nets = (): NetParams => ({ policy: policy.params, value: value.params, logStd })

// ---------------------------------------------------------------------------
// Checkpoints
// ---------------------------------------------------------------------------

function policyFile(stats: ReturnType<typeof summarise>): PolicyFile {
  return {
    version: 1,
    sizes: policy.sizes,
    params: encodeFloat32(policy.params),
    obsMean: Array.from(normalizer.mean),
    obsStd: Array.from(normalizer.std()),
    logStd: Array.from(logStd),
    meta: {
      iteration,
      envSteps,
      wallSeconds: Math.round(wallSeconds),
      meanReturn: stats.meanReturn,
      meanEpisodeSeconds: stats.meanSeconds,
      goalsPerEpisode: stats.goalsPerEpisode,
      fallRate: stats.fallRate,
      terrainLevel: LADDER[level].terrainLevel,
    },
  }
}

function saveState() {
  const s: TrainState = {
    iteration,
    envSteps,
    wallSeconds,
    level,
    lastPromotion,
    lr,
    policy: encodeFloat32(policy.params),
    value: encodeFloat32(value.params),
    logStd: Array.from(logStd),
    obs: { count: normalizer.count, mean: Array.from(normalizer.mean), m2: Array.from(normalizer.m2) },
  }
  writeFileSync(statePath, JSON.stringify(s))
}

// ---------------------------------------------------------------------------
// Statistics
// ---------------------------------------------------------------------------

let window: EpisodeStats[] = []
let recent: { goals: number; falls: number; steps: number }[] = []

function summarise(episodes: EpisodeStats[]) {
  const n = Math.max(1, episodes.length)
  const terms = Object.fromEntries(
    REWARD_TERMS.map((k) => [k, episodes.reduce((a, e) => a + e.terms[k], 0) / n]),
  ) as Record<string, number>
  return {
    episodes: episodes.length,
    meanReturn: episodes.reduce((a, e) => a + e.return, 0) / n,
    meanSeconds: episodes.reduce((a, e) => a + e.seconds, 0) / n,
    goalsPerEpisode: episodes.reduce((a, e) => a + e.goals, 0) / n,
    fallRate: episodes.filter((e) => e.fell).length / n,
    terms,
  }
}

// ---------------------------------------------------------------------------
// Training loop
// ---------------------------------------------------------------------------

await all(null, 'ready')
console.log(
  `training ${args.run}: ${numWorkers} workers × ${envsPerWorker} envs, ` +
    `${numWorkers * envsPerWorker * PPO.stepsPerEnv} samples/iter, obs ${OBS_SIZE}, act ${ACTION_SIZE}`,
)

while (iteration < maxIterations) {
  const t0 = performance.now()
  const curriculum = LADDER[level]

  // 1. Rollouts (observations normalised with the current statistics).
  const obsStd = normalizer.std()
  const obsMean = Float32Array.from(normalizer.mean)
  const rollouts = await all({ type: 'rollout', nets: nets(), obsMean, obsStd, curriculum }, 'rollout')
  let samples = 0
  const tally = { goals: 0, falls: 0, steps: 0 }
  for (const r of rollouts) {
    samples += r.samples
    tally.goals += r.goalsReached
    tally.falls += r.falls
    tally.steps += r.samples
    normalizer.merge(r.obsStats.count, r.obsStats.mean, r.obsStats.m2)
    window.push(...r.episodes)
  }
  envSteps += samples
  const rolloutMs = Math.max(...rollouts.map((r) => r.rolloutMs))

  // 2. Global advantage normalisation.
  const adv = await all({ type: 'advantageStats' }, 'advantageStats')
  const advN = adv.reduce((a, s) => a + s.n, 0)
  const advMean = adv.reduce((a, s) => a + s.sum, 0) / advN
  const advStd = Math.sqrt(Math.max(1e-8, adv.reduce((a, s) => a + s.sumSq, 0) / advN - advMean ** 2))

  // 3. PPO epochs over shuffled minibatches; gradients summed across workers.
  let kl = 0
  let clipFrac = 0
  let vLoss = 0
  let gradNorm = 0
  let updates = 0
  for (let epoch = 0; epoch < PPO.epochs; epoch++) {
    for (let mb = 0; mb < PPO.minibatches; mb++) {
      const grads = await all({ type: 'grad', nets: nets(), advMean, advStd, epoch, minibatch: mb }, 'grad')
      const n = grads.reduce((a, g) => a + g.n, 0)
      const gP = new Float32Array(policy.size)
      const gV = new Float32Array(value.size)
      const gS = new Float32Array(ACTION_SIZE)
      let klSum = 0
      for (const g of grads) {
        for (let i = 0; i < gP.length; i++) gP[i] += g.policy[i] / n
        for (let i = 0; i < gV.length; i++) gV[i] += g.value[i] / n
        for (let i = 0; i < gS.length; i++) gS[i] += g.logStd[i] / n
        klSum += g.klSum
        clipFrac += g.clipped / n
        vLoss += g.valueLoss / n
      }
      const mbKl = klSum / n
      // Adaptive learning rate on the approximate KL (as in rsl_rl).
      if (mbKl > PPO.desiredKl * 2) lr = Math.max(1e-5, lr / 1.5)
      else if (mbKl < PPO.desiredKl / 2 && mbKl > 0) lr = Math.min(1e-2, lr * 1.5)
      optPolicy.lr = optValue.lr = optLogStd.lr = lr

      gradNorm += clipGradNorm([gP, gS], PPO.maxGradNorm)
      clipGradNorm([gV], PPO.maxGradNorm)
      optPolicy.step(policy.params, gP)
      optLogStd.step(logStd, gS)
      optValue.step(value.params, gV)
      for (let i = 0; i < ACTION_SIZE; i++) logStd[i] = Math.min(0.5, Math.max(-2.5, logStd[i]))
      kl += mbKl
      updates++
    }
  }

  iteration++
  const iterSeconds = (performance.now() - t0) / 1000
  wallSeconds += iterSeconds

  // 4. Curriculum, logging, checkpoints.
  window = window.slice(-600)
  const stats = summarise(window)
  recent = [...recent, tally].slice(-RATE_WINDOW)
  const minutes = (recent.reduce((a, r) => a + r.steps, 0) * CONTROL_DT) / 60
  const goalsPerMinute = recent.reduce((a, r) => a + r.goals, 0) / minutes
  const fallsPerMinute = recent.reduce((a, r) => a + r.falls, 0) / minutes
  const settled = recent.length === RATE_WINDOW && iteration - lastPromotion >= PROMOTE.cooldownIterations
  if (settled && level < LADDER.length - 1 && goalsPerMinute >= PROMOTE.goalsPerMinute && fallsPerMinute <= PROMOTE.maxFallsPerMinute) {
    level++
    lastPromotion = iteration
    recent = []
    console.log(`  ▲ curriculum level ${level}: ${JSON.stringify(LADDER[level])}`)
  } else if (settled && level > 0 && fallsPerMinute > DEMOTE.fallsPerMinute) {
    level--
    lastPromotion = iteration
    recent = []
    console.log(`  ▼ curriculum level ${level} (too many falls): ${JSON.stringify(LADDER[level])}`)
  }

  const meanStd = logStd.reduce((a, s) => a + Math.exp(s), 0) / ACTION_SIZE
  const row = {
    iteration,
    envSteps,
    wallSeconds: Math.round(wallSeconds),
    level,
    ...stats,
    goalsPerMinute,
    fallsPerMinute,
    kl: kl / updates,
    clipFrac: clipFrac / updates,
    valueLoss: vLoss / updates,
    gradNorm: gradNorm / updates,
    lr,
    actionStd: meanStd,
    sps: Math.round(samples / iterSeconds),
    rolloutShare: rolloutMs / 1000 / iterSeconds,
  }
  appendFileSync(join(RUN_DIR, 'log.jsonl'), JSON.stringify(row) + '\n')
  console.log(
    `it ${String(iteration).padStart(5)} | ${(envSteps / 1e6).toFixed(2)}M steps | ${row.sps} sps | ` +
      `R ${stats.meanReturn.toFixed(2)} | ep ${stats.meanSeconds.toFixed(1)}s | goals ${stats.goalsPerEpisode.toFixed(2)} | ` +
      `fall ${(stats.fallRate * 100).toFixed(0)}% | ${goalsPerMinute.toFixed(1)} goals/min ${fallsPerMinute.toFixed(2)} falls/min | kl ${row.kl.toFixed(4)} lr ${lr.toExponential(1)} std ${meanStd.toFixed(2)} | L${level}`,
  )

  if (iteration % 10 === 0) {
    writeFileSync(join(RUN_DIR, 'latest.json'), JSON.stringify(policyFile(stats)))
    saveState()
  }
  if (iteration % 50 === 0) {
    writeFileSync(join(RUN_DIR, `iter_${String(iteration).padStart(5, '0')}.json`), JSON.stringify(policyFile(stats)))
  }
}

writeFileSync(join(RUN_DIR, 'latest.json'), JSON.stringify(policyFile(summarise(window))))
saveState()
await Promise.all(workers.map((w) => w.terminate()))
console.log('done')
