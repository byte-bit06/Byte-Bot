import { REWARD_TERMS, REWARDS, type RewardTerm } from './env'

/**
 * Recorded rollouts of a trained policy. The website plays these back instead
 * of running physics and the network live: same motion, a fraction of the cost.
 *
 * Poses are quantised to Int16 (positions in millimetres, quaternions × 32767)
 * and base64-encoded, so a 30 s clip of Byte's 24 parts is ~270 KB of JSON.
 */

export const REPLAY_HZ = 20
export const RATES_HZ = 5
export const POSE_STRIDE = 7 // px py pz qx qy qz qw

/** Continuous reward terms shown in the live bars (event terms are counted instead). */
export const RATE_TERMS = REWARD_TERMS.filter((k) => !(REWARDS[k] as { event?: boolean }).event)

export interface ClipEvent {
  /** Seconds from the start of the clip. */
  t: number
  type: 'goal' | 'fall'
}

export interface Clip {
  version: 1
  label: string
  terrain: { level: number; seed: number }
  seconds: number
  parts: number
  frames: number
  /** Base64 Int16: frames × parts × POSE_STRIDE. */
  poses: string
  /** Goal position from time `t` onwards. */
  goals: { t: number; x: number; z: number }[]
  events: ClipEvent[]
  /** Base64 Int16 (× 1000): reward rate per RATE_TERMS term, at RATES_HZ. */
  rates: string
}

export function encodeInt16(a: Int16Array): string {
  const bytes = new Uint8Array(a.buffer, a.byteOffset, a.byteLength)
  let s = ''
  for (let i = 0; i < bytes.length; i += 0x8000) s += String.fromCharCode(...bytes.subarray(i, i + 0x8000))
  return btoa(s)
}

export function decodeInt16(b64: string): Int16Array {
  const s = atob(b64)
  const bytes = new Uint8Array(s.length)
  for (let i = 0; i < s.length; i++) bytes[i] = s.charCodeAt(i)
  return new Int16Array(bytes.buffer)
}

const q16 = (v: number) => Math.max(-32767, Math.min(32767, Math.round(v)))

/** Builds a clip frame by frame while a policy runs. */
export class ClipRecorder {
  private readonly poses: number[] = []
  private readonly rates: number[] = []
  readonly goals: Clip['goals'] = []
  readonly events: ClipEvent[] = []
  frames = 0

  constructor(private readonly parts: number) {}

  pose(p: ArrayLike<{ x: number; y: number; z: number }>, q: ArrayLike<{ x: number; y: number; z: number; w: number }>) {
    for (let i = 0; i < this.parts; i++) {
      this.poses.push(q16(p[i].x * 1000), q16(p[i].y * 1000), q16(p[i].z * 1000))
      this.poses.push(q16(q[i].x * 32767), q16(q[i].y * 32767), q16(q[i].z * 32767), q16(q[i].w * 32767))
    }
    this.frames++
  }

  rate(rates: Record<RewardTerm, number>) {
    for (const k of RATE_TERMS) this.rates.push(q16(rates[k] * 1000))
  }

  goal(t: number, x: number, z: number) {
    const last = this.goals[this.goals.length - 1]
    if (!last || last.x !== x || last.z !== z) this.goals.push({ t: +t.toFixed(3), x: +x.toFixed(3), z: +z.toFixed(3) })
  }

  finish(label: string, terrain: Clip['terrain']): Clip {
    return {
      version: 1,
      label,
      terrain,
      seconds: this.frames / REPLAY_HZ,
      parts: this.parts,
      frames: this.frames,
      poses: encodeInt16(Int16Array.from(this.poses)),
      goals: this.goals,
      events: this.events,
      rates: encodeInt16(Int16Array.from(this.rates)),
    }
  }
}
