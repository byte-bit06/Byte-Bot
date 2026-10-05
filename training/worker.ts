/**
 * PPO worker thread: owns a slice of the environments, collects rollouts with
 * the current policy, computes GAE locally, and returns gradients for each
 * minibatch. Only parameters and gradients cross the thread boundary.
 */
import { parentPort, workerData } from 'node:worker_threads'
import RAPIER from '@dimforge/rapier3d-compat'
import { ACTION_SIZE, ByteEnv, CONTROL_DT, OBS_SIZE, REWARD_TERMS, type RewardTerms } from '../src/sim/env'
import { MLP, normalizeInto } from '../src/sim/policy'
import { createRng } from '../src/sim/random'
import { backwardBatch, forwardBatch } from './nn'
import { PPO, type EpisodeStats, type FromWorker, type ToWorker } from './protocol'

const { numEnvs, seed } = workerData as { numEnvs: number; seed: number }
const port = parentPort!
const send = (msg: FromWorker) => port.postMessage(msg)

await RAPIER.init()
const rng = createRng(seed)
const envs = Array.from({ length: numEnvs }, (_, i) => new ByteEnv(RAPIER, seed * 1000 + i))
const sizes = (out: number) => [OBS_SIZE, ...PPO.hidden, out]
const policy = new MLP(sizes(ACTION_SIZE))
const value = new MLP(sizes(1))
const logStd = new Float32Array(ACTION_SIZE)

// Current (un-normalised) observation of each env.
const obs = envs.map((env) => env.reset(new Float32Array(OBS_SIZE)))

// Rollout storage, sample index = t * numEnvs + e.
const T = PPO.stepsPerEnv
const N = T * numEnvs
const buf = {
  obs: new Float32Array(N * OBS_SIZE), // normalised, as fed to the networks
  actions: new Float32Array(N * ACTION_SIZE),
  logp: new Float32Array(N),
  values: new Float32Array(N),
  rewards: new Float32Array(N),
  dones: new Uint8Array(N),
  advantages: new Float32Array(N),
  returns: new Float32Array(N),
}

// Per-env running episode statistics.
const running = envs.map(() => ({ ret: 0, steps: 0, goals: 0, terms: zeroTerms() }))
function zeroTerms(): RewardTerms {
  return Object.fromEntries(REWARD_TERMS.map((k) => [k, 0])) as RewardTerms
}

const LOG_2PI = Math.log(2 * Math.PI)
function gaussianLogp(a: Float32Array, ao: number, mu: Float32Array, mo: number) {
  let lp = 0
  for (let i = 0; i < ACTION_SIZE; i++) {
    const s = Math.exp(logStd[i])
    const z = (a[ao + i] - mu[mo + i]) / s
    lp += -0.5 * z * z - logStd[i] - 0.5 * LOG_2PI
  }
  return lp
}

function rollout(msg: Extract<ToWorker, { type: 'rollout' }>) {
  const t0 = performance.now()
  policy.params.set(msg.nets.policy)
  value.params.set(msg.nets.value)
  logStd.set(msg.nets.logStd)
  for (const env of envs) env.curriculum = msg.curriculum

  // Welford stats over raw observations for the shared normaliser.
  const count = N
  const mean = new Float64Array(OBS_SIZE)
  const m2 = new Float64Array(OBS_SIZE)
  let seen = 0

  const norm = new Float32Array(OBS_SIZE)
  const mu = new Float32Array(ACTION_SIZE)
  const v = new Float32Array(1)
  const pScratch = policy.scratch()
  const vScratch = value.scratch()
  const action = new Float32Array(ACTION_SIZE)
  const episodes: EpisodeStats[] = []
  let goalsReached = 0
  let falls = 0

  for (let t = 0; t < T; t++) {
    for (let e = 0; e < numEnvs; e++) {
      const k = t * numEnvs + e
      const o = obs[e]
      seen++
      for (let i = 0; i < OBS_SIZE; i++) {
        const d = o[i] - mean[i]
        mean[i] += d / seen
        m2[i] += d * (o[i] - mean[i])
      }
      normalizeInto(o, msg.obsMean, msg.obsStd, norm)
      buf.obs.set(norm, k * OBS_SIZE)
      policy.forward(norm, mu, pScratch)
      value.forward(norm, v, vScratch)
      for (let i = 0; i < ACTION_SIZE; i++) action[i] = mu[i] + Math.exp(logStd[i]) * rng.normal()
      buf.actions.set(action, k * ACTION_SIZE)
      buf.logp[k] = gaussianLogp(buf.actions, k * ACTION_SIZE, mu, 0)
      buf.values[k] = v[0]

      const r = envs[e].step(action, o)
      let reward = r.reward
      const run = running[e]
      run.ret += r.reward
      run.steps++
      if (r.reachedGoal) {
        run.goals++
        goalsReached++
      }
      for (const key of REWARD_TERMS) run.terms[key] += r.terms[key]

      if (r.terminated) falls++
      if (r.truncated) {
        // Time limit is not a real terminal state: bootstrap from V(s').
        normalizeInto(o, msg.obsMean, msg.obsStd, norm)
        value.forward(norm, v, vScratch)
        reward += PPO.gamma * v[0]
      }
      buf.rewards[k] = reward
      buf.dones[k] = r.terminated || r.truncated ? 1 : 0
      if (buf.dones[k]) {
        episodes.push({
          return: run.ret,
          seconds: run.steps * CONTROL_DT,
          goals: run.goals,
          fell: r.terminated,
          terms: run.terms,
        })
        running[e] = { ret: 0, steps: 0, goals: 0, terms: zeroTerms() }
        envs[e].reset(o)
      }
    }
  }

  // GAE(λ), bootstrapping from the value of each env's current observation.
  const lastValue = new Float32Array(numEnvs)
  for (let e = 0; e < numEnvs; e++) {
    normalizeInto(obs[e], msg.obsMean, msg.obsStd, norm)
    lastValue[e] = value.forward(norm, v, vScratch)[0]
  }
  for (let e = 0; e < numEnvs; e++) {
    let gae = 0
    for (let t = T - 1; t >= 0; t--) {
      const k = t * numEnvs + e
      const notDone = 1 - buf.dones[k]
      const nextV = t === T - 1 ? lastValue[e] : buf.values[(t + 1) * numEnvs + e]
      const delta = buf.rewards[k] + PPO.gamma * nextV * notDone - buf.values[k]
      gae = delta + PPO.gamma * PPO.lambda * notDone * gae
      buf.advantages[k] = gae
      buf.returns[k] = gae + buf.values[k]
    }
  }

  send({
    type: 'rollout',
    samples: N,
    obsStats: { count, mean, m2 },
    episodes,
    goalsReached,
    falls,
    rolloutMs: performance.now() - t0,
  })
}

// A fresh shuffle per epoch, shared by all of that epoch's minibatches.
let perm = new Uint32Array(0)
let permEpoch = -1

function grad(msg: Extract<ToWorker, { type: 'grad' }>) {
  policy.params.set(msg.nets.policy)
  value.params.set(msg.nets.value)
  logStd.set(msg.nets.logStd)

  if (permEpoch !== msg.epoch || perm.length !== N) {
    perm = Uint32Array.from({ length: N }, (_, i) => i)
    for (let i = N - 1; i > 0; i--) {
      const j = rng.int(0, i + 1)
      ;[perm[i], perm[j]] = [perm[j], perm[i]]
    }
    permEpoch = msg.epoch
  }
  const size = Math.floor(N / PPO.minibatches)
  const idx = perm.subarray(msg.minibatch * size, (msg.minibatch + 1) * size)
  const n = idx.length

  const x = new Float32Array(n * OBS_SIZE)
  for (let s = 0; s < n; s++) x.set(buf.obs.subarray(idx[s] * OBS_SIZE, (idx[s] + 1) * OBS_SIZE), s * OBS_SIZE)

  const pc = forwardBatch(policy, x, n)
  const vc = forwardBatch(value, x, n)
  const dMu = new Float32Array(n * ACTION_SIZE)
  const dV = new Float32Array(n)
  const gLogStd = new Float32Array(ACTION_SIZE)
  let klSum = 0
  let clipped = 0
  let policyLoss = 0
  let valueLoss = 0

  for (let s = 0; s < n; s++) {
    const k = idx[s]
    const adv = (buf.advantages[k] - msg.advMean) / msg.advStd
    const logp = gaussianLogp(buf.actions, k * ACTION_SIZE, pc.output, s * ACTION_SIZE)
    const logRatio = logp - buf.logp[k]
    const ratio = Math.exp(logRatio)
    klSum += ratio - 1 - logRatio // low-variance KL estimator
    const surr = ratio * adv
    const clippedRatio = Math.min(1 + PPO.clip, Math.max(1 - PPO.clip, ratio))
    policyLoss += -Math.min(surr, clippedRatio * adv)
    const active = adv >= 0 ? ratio <= 1 + PPO.clip : ratio >= 1 - PPO.clip
    if (!active) clipped++
    else {
      // dLoss/dlogp = −ratio·A, chained through the Gaussian log-density.
      const g = -ratio * adv
      for (let i = 0; i < ACTION_SIZE; i++) {
        const sd = Math.exp(logStd[i])
        const z = (buf.actions[k * ACTION_SIZE + i] - pc.output[s * ACTION_SIZE + i]) / sd
        dMu[s * ACTION_SIZE + i] = (g * z) / sd
        gLogStd[i] += g * (z * z - 1)
      }
    }
    const err = vc.output[s] - buf.returns[k]
    valueLoss += 0.5 * err * err
    dV[s] = PPO.valueCoef * err
  }
  // Entropy bonus: H = Σ logσ + const per sample.
  for (let i = 0; i < ACTION_SIZE; i++) gLogStd[i] -= PPO.entropyCoef * n

  const gPolicy = new Float32Array(policy.size)
  const gValue = new Float32Array(value.size)
  backwardBatch(policy, pc, dMu, gPolicy)
  backwardBatch(value, vc, dV, gValue)
  send({ type: 'grad', n, policy: gPolicy, value: gValue, logStd: gLogStd, klSum, clipped, policyLoss, valueLoss })
}

port.on('message', (msg: ToWorker) => {
  if (msg.type === 'rollout') rollout(msg)
  else if (msg.type === 'advantageStats') {
    let sum = 0
    let sumSq = 0
    for (let i = 0; i < N; i++) {
      sum += buf.advantages[i]
      sumSq += buf.advantages[i] ** 2
    }
    send({ type: 'advantageStats', n: N, sum, sumSq })
  } else if (msg.type === 'grad') grad(msg)
})

send({ type: 'ready' })
