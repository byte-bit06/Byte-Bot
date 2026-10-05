import { create } from 'zustand'
import type { PolicyFile } from '../sim/policy'
import { REWARD_TERMS, type RewardTerm } from '../sim/env'

/** One point of the published learning curve. */
export interface CurvePoint {
  iteration: number
  envSteps: number
  meanReturn: number
  goalsPerEpisode: number
  fallRate: number
  episodeSeconds: number
  level: number
}

export interface CheckpointInfo {
  file: string
  /** Recorded playback of this policy (src/sim/replay.ts). */
  clip: string
  iteration: number
  envSteps: number
  wallSeconds: number
  meanReturn: number
  goalsPerEpisode: number
  fallRate: number
}

/** The final policy navigating on one terrain level, recorded. */
export interface NavigationClip {
  terrainLevel: number
  file: string
  goals: number
  falls: number
}

/** `public/policies/manifest.json`, written by `training/publish.ts`. */
export interface Manifest {
  run: string
  checkpoints: CheckpointInfo[]
  curve: CurvePoint[]
  navigation: NavigationClip[]
}

export interface LiveStats {
  /** Smoothed reward rate per term (reward per second). */
  rates: Record<RewardTerm, number>
  goals: number
  falls: number
  /** Seconds since the last fall or reset. */
  upFor: number
}

const zeroRates = () => Object.fromEntries(REWARD_TERMS.map((k) => [k, 0])) as Record<RewardTerm, number>

interface LabState {
  status: 'idle' | 'loading' | 'ready' | 'missing'
  manifest: Manifest | null
  /** Index into manifest.checkpoints of the policy driving Byte. */
  checkpoint: number
  /** Weights for the selected checkpoint; fetched only in live mode. */
  policy: PolicyFile | null
  /** Run the real network + physics in the browser instead of playing a recording. */
  liveMode: boolean
  terrainLevel: number
  live: LiveStats
  /** Goal placed by the visitor; `id` increments on every click. */
  goalRequest: { x: number; z: number; id: number } | null

  loadManifest: () => Promise<void>
  selectCheckpoint: (index: number) => Promise<void>
  setTerrainLevel: (level: number) => void
  setLiveMode: (on: boolean) => void
  setLive: (live: LiveStats) => void
  /** Visitor clicked the floor: send Byte there, switching to the live network if needed. */
  placeGoal: (x: number, z: number) => void
}

const cache = new Map<string, Promise<PolicyFile>>()
const fetchPolicy = (file: string) => {
  if (!cache.has(file)) cache.set(file, fetch(`${import.meta.env.BASE_URL}policies/${file}`).then((r) => r.json()))
  return cache.get(file)!
}

export const useLab = create<LabState>()((set, get) => ({
  status: 'idle',
  manifest: null,
  checkpoint: 0,
  policy: null,
  liveMode: false,
  terrainLevel: 0,
  live: { rates: zeroRates(), goals: 0, falls: 0, upFor: 0 },
  goalRequest: null,

  loadManifest: async () => {
    if (get().status !== 'idle') return
    set({ status: 'loading' })
    try {
      const res = await fetch(`${import.meta.env.BASE_URL}policies/manifest.json`)
      if (!res.ok) throw new Error(String(res.status))
      const manifest = (await res.json()) as Manifest
      set({ manifest, status: 'ready' })
      await get().selectCheckpoint(manifest.checkpoints.length - 1)
    } catch {
      set({ status: 'missing' })
    }
  },

  selectCheckpoint: async (index) => {
    const { manifest, liveMode } = get()
    if (!manifest) return
    set({ checkpoint: index })
    if (!liveMode) return
    const policy = await fetchPolicy(manifest.checkpoints[index].file)
    // Ignore stale loads if the visitor scrubbed on.
    if (get().checkpoint === index) set({ policy })
  },

  setLiveMode: (liveMode) => {
    set({ liveMode, live: freshLive(), ...(liveMode ? {} : { goalRequest: null }) })
    if (liveMode) void get().selectCheckpoint(get().checkpoint)
  },

  placeGoal: (x, z) => {
    set({ goalRequest: { x, z, id: (get().goalRequest?.id ?? 0) + 1 } })
    if (!get().liveMode) get().setLiveMode(true)
  },

  setTerrainLevel: (terrainLevel) => set({ terrainLevel, live: freshLive() }),
  setLive: (live) => set({ live }),
}))

export const freshLive = (): LiveStats => ({ rates: zeroRates(), goals: 0, falls: 0, upFor: 0 })
