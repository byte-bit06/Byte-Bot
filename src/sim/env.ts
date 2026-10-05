import { ByteSim, qRotate, type ContactState, type Rapier } from './byteSim'
import { createRng, type Rng } from './random'
import { makeTerrain, type Terrain } from './terrain'
import { WORLD } from '../robot/physics'

// ---------------------------------------------------------------------------
// Action space
// ---------------------------------------------------------------------------

/** Joints the policy controls. Neck, head and fingers are held at zero. */
export const ACTUATED = [
  'spine',
  'hip_yaw_L',
  'hip_roll_L',
  'hip_pitch_L',
  'knee_L',
  'ankle_L',
  'hip_yaw_R',
  'hip_roll_R',
  'hip_pitch_R',
  'knee_R',
  'ankle_R',
  'shoulder_L',
  'elbow_L',
  'shoulder_R',
  'elbow_R',
] as const
export const ACTION_SIZE = ACTUATED.length

/** Slightly crouched default stance; actions are offsets from it. */
const DEFAULT_POSE: Record<string, number> = {
  hip_pitch_L: -0.2,
  hip_pitch_R: -0.2,
  knee_L: 0.4,
  knee_R: 0.4,
  ankle_L: -0.2,
  ankle_R: -0.2,
  elbow_L: -0.3,
  elbow_R: -0.3,
}
const ACTION_SCALE: Record<string, number> = { spine: 0.2, hip_yaw_L: 0.3, hip_yaw_R: 0.3 }
const defaultOf = (id: string) => DEFAULT_POSE[id] ?? 0
const scaleOf = (id: string) => ACTION_SCALE[id] ?? 0.5

// ---------------------------------------------------------------------------
// Timing
// ---------------------------------------------------------------------------

/** Physics steps per policy step: 120 Hz physics, 40 Hz control. */
export const CONTROL_DECIMATION = 3
export const CONTROL_DT = WORLD.timestep * CONTROL_DECIMATION
export const EPISODE_SECONDS = 20
export const EPISODE_STEPS = Math.round(EPISODE_SECONDS / CONTROL_DT)
/** Period of the gait clock that drives the alternating-feet reward. */
export const GAIT_PERIOD = 0.7

// ---------------------------------------------------------------------------
// Observation space
// ---------------------------------------------------------------------------

export const OBS_LAYOUT = [
  ['gravity (waist frame)', 3],
  ['angular velocity (waist frame)', 3],
  ['linear velocity (waist frame)', 3],
  ['goal direction + distance', 3],
  ['joint positions', ACTION_SIZE],
  ['joint velocities', ACTION_SIZE],
  ['previous action', ACTION_SIZE],
  ['gait clock (sin, cos)', 2],
  ['foot contacts', 2],
] as const
export const OBS_SIZE = OBS_LAYOUT.reduce((n, [, size]) => n + size, 0)

// ---------------------------------------------------------------------------
// Rewards
// ---------------------------------------------------------------------------

/**
 * Reward terms. Continuous terms are rates (per second) and get multiplied by
 * the control timestep; `event` terms are one-off bonuses or penalties.
 */
export const REWARDS = {
  upright: { weight: 0.6, label: 'Chest up', note: 'exp(−tilt²/0.08) on the chest up-vector' },
  height: { weight: 0.3, label: 'Waist height', note: 'waist held ~0.58 m above the feet' },
  gait: { weight: 0.5, label: 'Foot pattern', note: 'stance matches the gait clock; both feet down when standing' },
  airTime: { weight: 1.0, label: 'Step length', note: 'bonus on touchdown for swings longer than 0.25 s' },
  progress: { weight: 1.5, label: 'Progress to goal', note: 'Δ distance to the green box per second' },
  heading: { weight: 0.3, label: 'Face the goal', note: 'cos(heading error) while walking' },
  stillness: { weight: 0.4, label: 'Stand still', note: 'low body speed while parked on the goal' },
  alive: { weight: 1.0, label: 'Alive', note: 'constant bonus for staying up (keeps survival net-positive)' },
  footSpacing: { weight: -3.0, label: 'Foot spacing', note: 'feet closer than 12 cm sideways (crossed legs)' },
  actionRate: { weight: -0.05, label: 'Smoothness', note: 'Σ(aₜ − aₜ₋₁)²' },
  jointVel: { weight: -0.0005, label: 'Joint speed', note: 'Σ q̇²' },
  jointLimits: { weight: -5.0, label: 'Joint limits', note: 'radians past 90% of a joint’s range' },
  wobble: { weight: -0.02, label: 'Body wobble', note: 'roll/pitch angular velocity²' },
  slip: { weight: -0.5, label: 'Foot slip', note: 'speed² of a foot that is on the ground' },
  goal: { weight: 10, label: 'Reached goal', note: 'one-off bonus inside the green box', event: true },
  fall: { weight: -10, label: 'Fell', note: 'one-off penalty; episode ends', event: true },
} as const satisfies Record<string, { weight: number; label: string; note: string; event?: boolean }>

export type RewardTerm = keyof typeof REWARDS
export type RewardTerms = Record<RewardTerm, number>
export const REWARD_TERMS = Object.keys(REWARDS) as RewardTerm[]

const zeroTerms = (): RewardTerms => Object.fromEntries(REWARD_TERMS.map((k) => [k, 0])) as RewardTerms

/** Inside this radius of the goal Byte counts as "on" it and should stand still. */
export const GOAL_RADIUS = 0.35
/** How long Byte must park on a goal before the next one appears. */
const GOAL_DWELL_SECONDS = 1.5
const NOMINAL_HEIGHT = 0.58

// ---------------------------------------------------------------------------
// Environment
// ---------------------------------------------------------------------------

export interface Curriculum {
  terrainLevel: number
  /** Goal spawn distance range, metres. */
  goalDistance: [number, number]
  /** Max angle of a new goal away from Byte's heading, radians. */
  goalAngle: number
  /** Domain randomisation strength in [0, 1]. */
  randomization: number
}

export const START_CURRICULUM: Curriculum = {
  terrainLevel: 0,
  goalDistance: [0.8, 1.5],
  goalAngle: 0.4,
  randomization: 0,
}

export interface StepResult {
  reward: number
  terms: RewardTerms
  /** Episode ended by failure. */
  terminated: boolean
  /** Episode ended by the time limit (bootstrap the value). */
  truncated: boolean
  reachedGoal: boolean
}

export interface Goal {
  x: number
  z: number
}

export class ByteEnv {
  sim: ByteSim
  terrain: Terrain
  goal: Goal = { x: 0, z: 1 }
  curriculum: Curriculum
  /** Set by the website: goals stay where the visitor puts them. */
  manualGoals = false
  /** Set when the caller chose the terrain (website, recordings, evaluation). */
  pinnedTerrain = false

  private readonly R: Rapier
  private rng: Rng
  private stepCount = 0
  private phase = 0
  private prevDist = 0
  private dwell = 0
  private goalAwarded = false
  private readonly action = new Float32Array(ACTION_SIZE)
  private readonly prevAction = new Float32Array(ACTION_SIZE)
  private contacts: ContactState = { footL: true, footR: true, body: false }
  private airTime = [0, 0]
  // Domain randomisation state.
  private motorScale = 1
  private delayed = false
  private pushTimer = 0

  constructor(R: Rapier, seed: number, curriculum: Curriculum = START_CURRICULUM) {
    this.R = R
    this.rng = createRng(seed)
    this.curriculum = curriculum
    this.terrain = makeTerrain(curriculum.terrainLevel, this.rng.int(0, 1e9))
    this.sim = new ByteSim(R, this.terrain)
  }

  /**
   * Start an episode. Training (auto goals) samples a terrain level up to the
   * curriculum maximum and sometimes a fresh layout, so the policy keeps flat
   * walking while learning rough ground. The website pins its own terrain.
   */
  reset(out: Float32Array): Float32Array {
    if (!this.manualGoals && !this.pinnedTerrain) {
      const level = this.rng.int(0, this.curriculum.terrainLevel + 1)
      if (level !== this.terrain.level || (level > 0 && this.rng.next() < 0.25))
        this.setTerrain(makeTerrain(level, this.rng.int(0, 1e9)))
    }
    const yaw = this.rng.range(-Math.PI, Math.PI)
    this.sim.reset(0, 0, yaw)
    this.stepCount = 0
    this.phase = this.rng.next()
    this.action.fill(0)
    this.prevAction.fill(0)
    this.airTime = [0, 0]
    this.contacts = { footL: true, footR: true, body: false }
    this.applyAction(this.action)

    const dr = this.curriculum.randomization
    this.motorScale = 1 + dr * this.rng.range(-0.15, 0.15)
    this.delayed = dr > 0 && this.rng.next() < 0.5 * dr
    this.pushTimer = this.rng.range(2, 5)
    this.sim.bodies.get('chest')!.setAdditionalMass(dr * this.rng.range(-1, 2), true)

    if (!this.manualGoals) this.spawnGoal(yaw)
    this.prevDist = this.goalDistance()
    return this.observe(out)
  }

  /** Swap the terrain. Callers outside training use `pinTerrain`. */
  setTerrain(terrain: Terrain) {
    // Rebuilding is simplest: the world owns all colliders.
    this.sim.free()
    this.terrain = terrain
    this.sim = new ByteSim(this.R, terrain)
  }

  /** Use this terrain for every episode (website, recordings, evaluation). */
  pinTerrain(terrain: Terrain) {
    this.pinnedTerrain = true
    this.setTerrain(terrain)
  }

  setGoal(goal: Goal) {
    this.goal = goal
    this.goalAwarded = false
    this.dwell = 0
    this.prevDist = this.goalDistance()
  }

  private spawnGoal(heading: number) {
    const [lo, hi] = this.curriculum.goalDistance
    const d = this.rng.range(lo, hi)
    const a = heading + this.rng.range(-this.curriculum.goalAngle, this.curriculum.goalAngle)
    const p = this.sim.bodies.get('waist')!.translation()
    this.setGoal({ x: p.x + Math.sin(a) * d, z: p.z + Math.cos(a) * d })
  }

  goalDistance(): number {
    const p = this.sim.bodies.get('waist')!.translation()
    return Math.hypot(this.goal.x - p.x, this.goal.z - p.z)
  }

  private applyAction(a: Float32Array) {
    for (let i = 0; i < ACTION_SIZE; i++) {
      const id = ACTUATED[i]
      this.sim.drive(id, defaultOf(id) + scaleOf(id) * a[i], this.motorScale)
    }
  }

  step(rawAction: Float32Array, out: Float32Array): StepResult {
    const { sim } = this
    this.prevAction.set(this.action)
    for (let i = 0; i < ACTION_SIZE; i++) this.action[i] = Math.max(-1, Math.min(1, rawAction[i]))
    // Simulated actuation latency: apply last step's command this step.
    this.applyAction(this.delayed ? this.prevAction : this.action)

    this.maybePush()
    for (let i = 0; i < CONTROL_DECIMATION; i++) sim.step()
    this.stepCount++
    this.phase = (this.phase + CONTROL_DT / GAIT_PERIOD) % 1

    const terms = zeroTerms()
    const dt = CONTROL_DT
    const waist = sim.bodies.get('waist')!
    const chest = sim.bodies.get('chest')!
    const wp = waist.translation()
    const contacts = sim.contacts()
    const touchdown = [contacts.footL && !this.contacts.footL, contacts.footR && !this.contacts.footR]
    this.contacts = contacts

    // --- Posture ------------------------------------------------------------
    const chestUp = qRotate(chest.rotation(), { x: 0, y: 1, z: 0 })
    const tilt = Math.acos(Math.max(-1, Math.min(1, chestUp.y)))
    terms.upright = Math.exp(-(tilt * tilt) / 0.08)
    const footL = sim.bodies.get('foot_L')!.translation()
    const footR = sim.bodies.get('foot_R')!.translation()
    const height = wp.y - Math.min(footL.y, footR.y)
    terms.height = Math.exp(-((height - NOMINAL_HEIGHT) ** 2) / 0.004)

    // --- Goal ---------------------------------------------------------------
    const dist = this.goalDistance()
    const parked = dist < GOAL_RADIUS
    let reachedGoal = false
    if (parked) {
      if (!this.goalAwarded) {
        this.goalAwarded = true
        reachedGoal = true
        terms.goal = 1
      }
      this.dwell += dt
      const v = waist.linvel()
      terms.stillness = Math.exp(-(v.x * v.x + v.z * v.z) / 0.02)
    } else {
      terms.progress = Math.max(-1, Math.min(1, (this.prevDist - dist) / dt))
      const fwd = qRotate(waist.rotation(), { x: 0, y: 0, z: 1 })
      const toGoal = { x: (this.goal.x - wp.x) / dist, z: (this.goal.z - wp.z) / dist }
      const flat = Math.hypot(fwd.x, fwd.z) || 1
      terms.heading = (fwd.x * toGoal.x + fwd.z * toGoal.z) / flat
    }
    this.prevDist = dist

    // --- Feet ---------------------------------------------------------------
    const swing = Math.sin(2 * Math.PI * this.phase)
    const expectL = parked || swing > -0.3 // left stance in the first half-cycle
    const expectR = parked || swing < 0.3
    terms.gait = ((expectL === contacts.footL ? 1 : 0) + (expectR === contacts.footR ? 1 : 0)) / 2
    const feet = [contacts.footL, contacts.footR]
    for (let f = 0; f < 2; f++) {
      if (touchdown[f] && !parked) terms.airTime += Math.max(-0.05, Math.min(this.airTime[f] - 0.25, 0.3)) / dt
      this.airTime[f] = feet[f] ? 0 : this.airTime[f] + dt
    }
    // Sideways foot separation in Byte's heading frame.
    const right = qRotate(waist.rotation(), { x: 1, y: 0, z: 0 })
    const side = Math.abs((footL.x - footR.x) * right.x + (footL.z - footR.z) * right.z)
    terms.footSpacing = Math.max(0, 0.12 - side) / 0.12
    for (const [f, id] of [
      [contacts.footL, 'foot_L'],
      [contacts.footR, 'foot_R'],
    ] as const) {
      if (!f) continue
      const v = sim.bodies.get(id)!.linvel()
      terms.slip += v.x * v.x + v.z * v.z
    }

    // --- Regularisers -------------------------------------------------------
    terms.alive = 1
    for (let i = 0; i < ACTION_SIZE; i++) {
      const d = this.action[i] - this.prevAction[i]
      terms.actionRate += d * d
      const id = ACTUATED[i]
      const qv = sim.jointVelocity(id)
      terms.jointVel += qv * qv
      const [lo, hi] = sim.jointDefs.get(id)!.limits
      const q = sim.jointAngle(id)
      const margin = 0.05 * (hi - lo)
      terms.jointLimits += Math.max(0, lo + margin - q) + Math.max(0, q - (hi - margin))
    }
    const wq = waist.rotation()
    const w = qRotate({ x: -wq.x, y: -wq.y, z: -wq.z, w: wq.w }, waist.angvel())
    terms.wobble = w.x * w.x + w.z * w.z

    // --- Termination --------------------------------------------------------
    const terminated = contacts.body || height < 0.3 || tilt > 1.0
    if (terminated) terms.fall = 1
    const truncated = !terminated && this.stepCount >= EPISODE_STEPS

    let reward = 0
    for (const k of REWARD_TERMS) {
      const spec = REWARDS[k] as { weight: number; event?: boolean }
      terms[k] *= spec.weight * (spec.event ? 1 : dt)
      reward += terms[k]
    }

    // Next goal once Byte has parked on this one for a moment.
    if (parked && this.dwell >= GOAL_DWELL_SECONDS && !this.manualGoals) {
      const fwd = qRotate(waist.rotation(), { x: 0, y: 0, z: 1 })
      this.spawnGoal(Math.atan2(fwd.x, fwd.z))
    }

    this.observe(out)
    return { reward, terms, terminated, truncated, reachedGoal }
  }

  private maybePush() {
    const dr = this.curriculum.randomization
    if (dr <= 0) return
    this.pushTimer -= CONTROL_DT
    if (this.pushTimer > 0) return
    this.pushTimer = this.rng.range(3, 6)
    const a = this.rng.range(0, Math.PI * 2)
    const j = this.rng.range(0, 25) * dr
    this.sim.bodies.get('waist')!.applyImpulse({ x: Math.cos(a) * j, y: 0, z: Math.sin(a) * j }, true)
  }

  /** Fill `out` with the current observation (see OBS_LAYOUT). */
  observe(out: Float32Array): Float32Array {
    const { sim } = this
    const waist = sim.bodies.get('waist')!
    const q = waist.rotation()
    const qi = { x: -q.x, y: -q.y, z: -q.z, w: q.w }
    const noise = this.curriculum.randomization
    let o = 0

    const g = qRotate(qi, { x: 0, y: -1, z: 0 })
    out[o++] = g.x
    out[o++] = g.y
    out[o++] = g.z
    const w = qRotate(qi, waist.angvel())
    out[o++] = w.x * 0.25
    out[o++] = w.y * 0.25
    out[o++] = w.z * 0.25
    const v = qRotate(qi, waist.linvel())
    out[o++] = v.x
    out[o++] = v.y
    out[o++] = v.z

    // Goal in the heading (yaw-only) frame, so it is independent of body tilt.
    const fwd = qRotate(q, { x: 0, y: 0, z: 1 })
    const yaw = Math.atan2(fwd.x, fwd.z)
    const p = waist.translation()
    const dx = this.goal.x - p.x
    const dz = this.goal.z - p.z
    const c = Math.cos(-yaw)
    const s = Math.sin(-yaw)
    const lx = c * dx + s * dz
    const lz = -s * dx + c * dz
    const d = Math.hypot(lx, lz)
    const k = d > 3 ? 3 / d : 1
    out[o++] = (lx * k) / 3
    out[o++] = (lz * k) / 3
    out[o++] = Math.min(d, 5) / 5

    for (let i = 0; i < ACTION_SIZE; i++)
      out[o++] = sim.jointAngle(ACTUATED[i]) - defaultOf(ACTUATED[i]) + noise * 0.01 * this.rng.normal()
    for (let i = 0; i < ACTION_SIZE; i++)
      out[o++] = sim.jointVelocity(ACTUATED[i]) * 0.05 + noise * 0.05 * this.rng.normal()
    for (let i = 0; i < ACTION_SIZE; i++) out[o++] = this.action[i]

    out[o++] = Math.sin(2 * Math.PI * this.phase)
    out[o++] = Math.cos(2 * Math.PI * this.phase)
    out[o++] = this.contacts.footL ? 1 : 0
    out[o++] = this.contacts.footR ? 1 : 0
    return out
  }

  free() {
    this.sim.free()
  }
}
