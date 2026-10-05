import { useEffect, useMemo, useRef, useState } from 'react'
import { useFrame } from '@react-three/fiber'
import type { Group, MeshStandardMaterial } from 'three'
import { REWARD_TERMS, type RewardTerm } from '../sim/env'
import { makeTerrain } from '../sim/terrain'
import { POSE_STRIDE, RATE_TERMS, RATES_HZ, REPLAY_HZ, decodeInt16, type Clip } from '../sim/replay'
import { freshLive, useLab } from '../story/lab'
import { ByteRig, TerrainBlocks, WAIST, useFollowCamera } from './ByteRig'
import { GoalBox } from './GoalBox'

const STATS_HZ = 6

interface DecodedClip {
  clip: Clip
  poses: Int16Array
  rates: Int16Array
}

const cache = new Map<string, Promise<DecodedClip>>()
function loadClip(file: string) {
  if (!cache.has(file)) {
    cache.set(
      file,
      fetch(`${import.meta.env.BASE_URL}policies/${file}`)
        .then((r) => r.json() as Promise<Clip>)
        .then((clip) => ({ clip, poses: decodeInt16(clip.poses), rates: decodeInt16(clip.rates) })),
    )
  }
  return cache.get(file)!
}

/**
 * Plays back a recorded rollout of a trained policy (see training/record.ts).
 * No physics or neural network runs here — just interpolation between
 * 20 Hz keyframes — so it stays smooth on any device.
 */
export function ReplayRobot({ file }: { file: string }) {
  const [data, setData] = useState<DecodedClip | null>(null)
  useEffect(() => {
    let live = true
    void loadClip(file).then((d) => live && setData(d))
    return () => {
      live = false
    }
  }, [file])
  return data ? <Playback key={file} data={data} /> : null
}

function Playback({ data }: { data: DecodedClip }) {
  const { clip, poses, rates } = data
  const terrain = useMemo(() => makeTerrain(clip.terrain.level, clip.terrain.seed), [clip])
  const setLive = useLab((s) => s.setLive)
  const follow = useFollowCamera()
  const parts = useRef<(Group | null)[]>([])
  const goalRef = useRef<Group>(null)
  const goalMat = useRef<MeshStandardMaterial>(null)
  const state = useMemo(() => ({ t: 0, flash: 0, statsTimer: 0, live: freshLive(), cursor: 0 }), [])

  useFrame((_, delta) => {
    const dt = Math.min(delta, 0.1)
    state.t += dt
    if (state.t >= clip.seconds) {
      // Loop the clip and restart its counters.
      state.t = 0
      state.cursor = 0
      state.live = freshLive()
    }
    const t = state.t

    // Events that happened since the last frame.
    while (state.cursor < clip.events.length && clip.events[state.cursor].t <= t) {
      const e = clip.events[state.cursor++]
      if (e.type === 'goal') {
        state.live.goals++
        state.flash = 1
      } else {
        state.live.falls++
        state.live.upFor = 0
      }
    }
    state.live.upFor += dt

    // Interpolate between keyframes.
    const f = t * REPLAY_HZ
    const f0 = Math.min(clip.frames - 1, Math.floor(f))
    const f1 = Math.min(clip.frames - 1, f0 + 1)
    const a = f - f0
    const stride = clip.parts * POSE_STRIDE
    parts.current.forEach((g, i) => {
      if (!g) return
      const o0 = f0 * stride + i * POSE_STRIDE
      const o1 = f1 * stride + i * POSE_STRIDE
      g.position.set(
        (poses[o0] + (poses[o1] - poses[o0]) * a) / 1000,
        (poses[o0 + 1] + (poses[o1 + 1] - poses[o0 + 1]) * a) / 1000,
        (poses[o0 + 2] + (poses[o1 + 2] - poses[o0 + 2]) * a) / 1000,
      )
      // A short nlerp of unit quaternions is visually identical to slerp here.
      const sign = poses[o0 + 3] * poses[o1 + 3] + poses[o0 + 4] * poses[o1 + 4] + poses[o0 + 5] * poses[o1 + 5] + poses[o0 + 6] * poses[o1 + 6] < 0 ? -1 : 1
      g.quaternion
        .set(
          poses[o0 + 3] + (sign * poses[o1 + 3] - poses[o0 + 3]) * a,
          poses[o0 + 4] + (sign * poses[o1 + 4] - poses[o0 + 4]) * a,
          poses[o0 + 5] + (sign * poses[o1 + 5] - poses[o0 + 5]) * a,
          poses[o0 + 6] + (sign * poses[o1 + 6] - poses[o0 + 6]) * a,
        )
        .normalize()
    })

    // Current goal.
    let goal = clip.goals[0]
    for (const g of clip.goals) if (g.t <= t) goal = g
    goalRef.current?.position.set(goal.x, 0, goal.z)
    state.flash = Math.max(0, state.flash - dt * 1.5)
    if (goalMat.current) goalMat.current.emissiveIntensity = 0.6 + 2.5 * state.flash

    follow(parts.current[WAIST], dt)

    state.statsTimer += dt
    if (state.statsTimer > 1 / STATS_HZ) {
      state.statsTimer = 0
      const r = Math.min(Math.floor(t * RATES_HZ), rates.length / RATE_TERMS.length - 1)
      const live = { ...state.live, rates: { ...state.live.rates } }
      for (const k of REWARD_TERMS) live.rates[k as RewardTerm] = 0
      RATE_TERMS.forEach((k, i) => (live.rates[k] = rates[r * RATE_TERMS.length + i] / 1000))
      setLive(live)
    }
  })

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
