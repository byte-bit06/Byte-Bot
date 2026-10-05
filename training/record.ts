/**
 * Record rollouts of a policy for website playback (see src/sim/replay.ts).
 * Mirrors the site's live loop exactly: 40 Hz control, a 1.5 s pause after a
 * fall, then Byte stands back up and carries on toward the same goal.
 */
import type RAPIER_NS from '@dimforge/rapier3d-compat'
import { ACTION_SIZE, ByteEnv, CONTROL_DT, OBS_SIZE, REWARD_TERMS, type RewardTerm } from '../src/sim/env'
import { makeTerrain } from '../src/sim/terrain'
import { Policy, type PolicyFile } from '../src/sim/policy'
import { ClipRecorder, RATES_HZ, REPLAY_HZ, type Clip } from '../src/sim/replay'
import { PARTS } from '../src/robot/skeleton'

const FALL_RESET_SECONDS = 1.5
const STEPS_PER_FRAME = Math.round(1 / CONTROL_DT / REPLAY_HZ)
const STEPS_PER_RATE = Math.round(1 / CONTROL_DT / RATES_HZ)

export interface RecordOptions {
  label: string
  seconds: number
  terrainLevel: number
  terrainSeed: number
  seed: number
}

export function recordClip(R: typeof RAPIER_NS, file: PolicyFile | null, o: RecordOptions) {
  const terrain = makeTerrain(o.terrainLevel, o.terrainSeed)
  const env = new ByteEnv(R, o.seed, {
    terrainLevel: o.terrainLevel,
    goalDistance: [1.2, 2.8],
    goalAngle: 1.2,
    randomization: 0,
  })
  env.pinTerrain(terrain)
  const policy = file ? new Policy(file) : null
  const obs = env.reset(new Float32Array(OBS_SIZE))
  const action = new Float32Array(ACTION_SIZE)
  const rec = new ClipRecorder(PARTS.length)
  const rates = Object.fromEntries(REWARD_TERMS.map((k) => [k, 0])) as Record<RewardTerm, number>
  const k = 1 - Math.exp(-CONTROL_DT / 1.5)
  let goals = 0
  let falls = 0
  let downFor = -1

  const totalSteps = Math.round(o.seconds / CONTROL_DT)
  for (let step = 0; step < totalSteps; step++) {
    const t = step * CONTROL_DT
    if (step % STEPS_PER_FRAME === 0) {
      const bodies = PARTS.map((p) => env.sim.bodies.get(p.id)!)
      rec.pose(
        bodies.map((b) => b.translation()),
        bodies.map((b) => b.rotation()),
      )
      rec.goal(t, env.goal.x, env.goal.z)
    }
    if (step % STEPS_PER_RATE === 0) rec.rate(rates)

    if (downFor >= 0) {
      downFor += CONTROL_DT
      for (let i = 0; i < 3; i++) env.sim.step()
      if (downFor > FALL_RESET_SECONDS) {
        const goal = env.goal
        env.reset(obs)
        env.setGoal(goal)
        downFor = -1
      }
      continue
    }
    if (policy) policy.act(obs, action)
    else action.fill(0)
    const r = env.step(action, obs)
    for (const term of REWARD_TERMS) rates[term] += (r.terms[term] / CONTROL_DT - rates[term]) * k
    if (r.reachedGoal) {
      goals++
      rec.events.push({ t: +t.toFixed(3), type: 'goal' })
    }
    if (r.terminated) {
      falls++
      downFor = 0
      rec.events.push({ t: +t.toFixed(3), type: 'fall' })
    }
  }
  env.free()
  const clip: Clip = rec.finish(o.label, { level: o.terrainLevel, seed: o.terrainSeed })
  return { clip, goals, falls }
}

/** Record a few seeds and keep the most representative of the policy at its best. */
export function recordBest(R: typeof RAPIER_NS, file: PolicyFile | null, o: Omit<RecordOptions, 'seed'>, tries = 4) {
  let best: ReturnType<typeof recordClip> | null = null
  for (let s = 0; s < tries; s++) {
    const run = recordClip(R, file, { ...o, seed: 1000 + s })
    if (!best || run.goals - 2 * run.falls > best.goals - 2 * best.falls) best = run
  }
  return best!
}
