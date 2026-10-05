import { useEffect, useMemo, useRef } from 'react'
import { useFrame } from '@react-three/fiber'
import RAPIER from '@dimforge/rapier3d-compat'
import { Quaternion, Vector3, type Group, type MeshStandardMaterial } from 'three'
import { ACTION_SIZE, ByteEnv, CONTROL_DT, OBS_SIZE, REWARD_TERMS } from '../sim/env'
import { makeTerrain, type Terrain } from '../sim/terrain'
import { Policy, type PolicyFile } from '../sim/policy'
import { PARTS } from '../robot/skeleton'
import { freshLive, useLab } from '../story/lab'
import { ByteRig, TerrainBlocks, WAIST, useFollowCamera } from './ByteRig'
import { GoalBox } from './GoalBox'

interface PolicyRobotProps {
  /** Trained policy; `null` sends zero actions (an untrained, rigid Byte). */
  policyFile: PolicyFile | null
  terrainLevel: number
}

const FALL_RESET_SECONDS = 1.5
const MAX_STEPS_PER_FRAME = 4
const STATS_HZ = 6
const TERRAIN_SEED = 2024

type Pose = { p: Vector3; q: Quaternion }[]

/**
 * Byte driven by a trained neural-network policy, live in the browser.
 *
 * Runs the exact environment used for training (same physics, observations
 * and rewards) in its own Rapier world, steps it in real time at the 40 Hz
 * control rate, and interpolates the rendered pose between control steps.
 *
 * Only the mount effect and the frame loop touch the Rapier world; rendering
 * works from plain data. That keeps a freed world unreachable under React's
 * StrictMode double-mount and concurrent rendering.
 */
export function PolicyRobot({ policyFile, terrainLevel }: PolicyRobotProps) {
  const terrain = useMemo(() => makeTerrain(terrainLevel, TERRAIN_SEED), [terrainLevel])
  const policy = useMemo(() => (policyFile ? new Policy(policyFile) : null), [policyFile])
  const setLive = useLab((s) => s.setLive)
  const follow = useFollowCamera()

  const envRef = useRef<ByteEnv | null>(null)
  const parts = useRef<(Group | null)[]>([])
  const goalRef = useRef<Group>(null)
  const goalMat = useRef<MeshStandardMaterial>(null)

  const sim = useMemo(
    () => ({
      obs: new Float32Array(OBS_SIZE),
      action: new Float32Array(ACTION_SIZE),
      acc: 0,
      downFor: -1, // seconds since a fall, or −1 while upright
      flash: 0,
      prev: PARTS.map(() => ({ p: new Vector3(), q: new Quaternion() })) as Pose,
      curr: PARTS.map(() => ({ p: new Vector3(), q: new Quaternion() })) as Pose,
      live: freshLive(),
      statsTimer: 0,
    }),
    [],
  )

  useEffect(() => {
    const env = createEnv(terrain)
    env.reset(sim.obs)
    snapshot(env, sim.curr)
    snapshot(env, sim.prev)
    Object.assign(sim, { acc: 0, downFor: -1, flash: 0, live: freshLive() })
    envRef.current = env
    applyGoalRequest(env)
    return () => {
      envRef.current = null
      env.free()
    }
  }, [terrain, sim])

  // Goals the visitor places stay put: Byte walks there and waits for the next click.
  const goalRequest = useLab((s) => s.goalRequest)
  useEffect(() => {
    if (envRef.current) applyGoalRequest(envRef.current)
  }, [goalRequest])

  useFrame((_, delta) => {
    const env = envRef.current
    if (!env) return
    const dt = Math.min(delta, 0.1)
    sim.acc += dt
    let steps = 0
    while (sim.acc >= CONTROL_DT && steps < MAX_STEPS_PER_FRAME) {
      sim.acc -= CONTROL_DT
      steps++
      sim.prev.forEach((s, i) => {
        s.p.copy(sim.curr[i].p)
        s.q.copy(sim.curr[i].q)
      })
      if (sim.downFor >= 0) recoverFromFall(env)
      else controlStep(env)
      snapshot(env, sim.curr)
    }
    if (steps === MAX_STEPS_PER_FRAME) sim.acc = 0

    // Interpolate the rendered pose between the last two control steps.
    const alpha = Math.min(1, sim.acc / CONTROL_DT)
    parts.current.forEach((g, i) => {
      if (!g) return
      g.position.lerpVectors(sim.prev[i].p, sim.curr[i].p, alpha)
      g.quaternion.slerpQuaternions(sim.prev[i].q, sim.curr[i].q, alpha)
    })

    sim.flash = Math.max(0, sim.flash - dt * 1.5)
    goalRef.current?.position.set(env.goal.x, 0, env.goal.z)
    if (goalMat.current) goalMat.current.emissiveIntensity = 0.6 + 2.5 * sim.flash

    follow(parts.current[WAIST], dt)

    sim.statsTimer += dt
    if (sim.statsTimer > 1 / STATS_HZ) {
      sim.statsTimer = 0
      setLive({ ...sim.live, rates: { ...sim.live.rates } })
    }
  })

  function controlStep(env: ByteEnv) {
    if (policy) policy.act(sim.obs, sim.action)
    else sim.action.fill(0)
    const r = env.step(sim.action, sim.obs)
    const k = 1 - Math.exp(-CONTROL_DT / 1.5) // ~1.5 s moving average
    for (const term of REWARD_TERMS) sim.live.rates[term] += (r.terms[term] / CONTROL_DT - sim.live.rates[term]) * k
    sim.live.upFor += CONTROL_DT
    if (r.reachedGoal) {
      sim.live.goals++
      sim.flash = 1
    }
    // Episode time limits only matter in training; on the site Byte keeps going.
    if (r.terminated) {
      sim.live.falls++
      sim.downFor = 0
    }
  }

  /** Lie still for a moment after a fall, then stand back up. */
  function recoverFromFall(env: ByteEnv) {
    sim.downFor += CONTROL_DT
    for (let i = 0; i < 3; i++) env.sim.step()
    if (sim.downFor <= FALL_RESET_SECONDS) return
    const goal = env.goal
    env.reset(sim.obs)
    env.setGoal(goal)
    sim.downFor = -1
    sim.live.upFor = 0
    snapshot(env, sim.prev)
  }

  return (
    <group>
      <TerrainBlocks terrain={terrain} />

      <group ref={goalRef}>
        <GoalBox materialRef={goalMat} />
      </group>

      <ByteRig parts={parts} />
    </group>
  )
}

function applyGoalRequest(env: ByteEnv) {
  const request = useLab.getState().goalRequest
  if (!request) return
  env.manualGoals = true
  env.setGoal({ x: request.x, z: request.z })
}

function createEnv(terrain: Terrain) {
  const env = new ByteEnv(RAPIER, 12345, {
    terrainLevel: terrain.level,
    goalDistance: [1.2, 2.8],
    goalAngle: 1.2,
    randomization: 0,
  })
  env.pinTerrain(terrain)
  return env
}

function snapshot(env: ByteEnv, into: Pose) {
  PARTS.forEach((part, i) => {
    const b = env.sim.bodies.get(part.id)!
    const t = b.translation()
    const r = b.rotation()
    into[i].p.set(t.x, t.y, t.z)
    into[i].q.set(r.x, r.y, r.z, r.w)
  })
}
