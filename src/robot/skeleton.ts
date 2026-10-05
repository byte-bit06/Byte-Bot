import { Quaternion, Vector3 } from 'three'

/**
 * Byte's body plan.
 *
 * Every body part is built from primitive "beans" (capsules, or balls for the
 * head and hip housings),
 * defined in its rest pose: standing upright, facing +Z, feet on the floor at
 * y = 0. Byte's left side is +X.
 *
 * All rigid bodies rest with an identity rotation; the capsule's orientation
 * lives on the collider and mesh instead. That keeps joint anchors and axes
 * identical in world and body-local space, which makes the joint table below
 * easy to read and edit.
 */

export type V3 = [number, number, number]
export type Q4 = [number, number, number, number]
export type Finish = 'shell' | 'graphite' | 'accent'

/** One collider + mesh, positioned relative to its part's origin. */
export type PartShape =
  | { kind: 'capsule'; radius: number; halfHeight: number; quaternion: Q4; offset: V3 }
  | { kind: 'ball'; radius: number; offset: V3 }

export interface PartDef {
  id: string
  label: string
  /** Rest-pose origin of the rigid body, world space. */
  center: V3
  /** Most parts are a single bean; feet are two toes side by side. */
  shapes: PartShape[]
  /** Collider density (kg/m³). Mass and inertia are derived from it. */
  density: number
  finish: Finish
}

export interface JointDef {
  id: string
  label: string
  parent: string
  child: string
  /** Pivot in each body's local frame. */
  parentAnchor: V3
  childAnchor: V3
  /** Hinge axis (identical in world and local frames at rest). */
  axis: V3
  /** Angular limits in radians, measured child-relative-to-parent. */
  limits: [number, number]
  /** Scales motor stiffness/damping: legs carry the body, fingers don't. */
  strength: number
}

const Y_UP = new Vector3(0, 1, 0)
const toV3 = (v: Vector3): V3 => [v.x, v.y, v.z]
const sub = (a: V3, b: V3): V3 => [a[0] - b[0], a[1] - b[1], a[2] - b[2]]

/**
 * A capsule spanning the segment between two points. The tips stop `gap`
 * short of each end so joint markers stay visible between beans.
 */
function capsule(from: V3, to: V3, radius: number, gap: number, origin: V3): PartShape {
  const a = new Vector3(...from)
  const b = new Vector3(...to)
  const dir = b.clone().sub(a)
  const length = dir.length()
  const q = new Quaternion().setFromUnitVectors(Y_UP, dir.normalize())
  return {
    kind: 'capsule',
    radius,
    halfHeight: Math.max(0.004, length / 2 - radius - gap),
    quaternion: [q.x, q.y, q.z, q.w],
    offset: sub(toV3(a.add(b).multiplyScalar(0.5)), origin),
  }
}

interface PartOpts {
  density?: number
  finish?: Finish
  gap?: number
}

/** A single-bean part whose origin is the bean's midpoint. */
function bean(id: string, label: string, from: V3, to: V3, radius: number, opts: PartOpts = {}): PartDef {
  const center = toV3(new Vector3(...from).add(new Vector3(...to)).multiplyScalar(0.5))
  return {
    id,
    label,
    center,
    shapes: [capsule(from, to, radius, opts.gap ?? 0.02, center)],
    density: opts.density ?? 1000,
    finish: opts.finish ?? 'shell',
  }
}

function ball(id: string, label: string, center: V3, radius: number, opts: PartOpts = {}): PartDef {
  return {
    id,
    label,
    center,
    shapes: [{ kind: 'ball', radius, offset: [0, 0, 0] }],
    density: opts.density ?? 1000,
    finish: opts.finish ?? 'shell',
  }
}

// ---------------------------------------------------------------------------
// Landmarks (world space, rest pose)
// ---------------------------------------------------------------------------

const SIDES = [
  { tag: 'L', name: 'Left', s: 1 },
  { tag: 'R', name: 'Right', s: -1 },
] as const

const SPINE: V3 = [0, 0.72, 0]
const NECK: V3 = [0, 1.07, 0]
const HEAD: V3 = [0, 1.165, 0]

// Each hip is three hinges in series (yaw → roll → pitch) linked by small,
// dense "actuator housings". Rapier only motorizes single-axis joints, and a
// light link between heavy bodies destabilises the solver, hence the density.
const HIP_YAW = (s: number): V3 => [0.1 * s, 0.6, 0]
const HIP_ROLL = (s: number): V3 => [0.1 * s, 0.55, 0]
const HIP_PITCH = (s: number): V3 => [0.1 * s, 0.52, 0]
const KNEE = (s: number): V3 => [0.1 * s, 0.3, 0]
const ANKLE = (s: number): V3 => [0.1 * s, 0.09, 0]
const SHOULDER = (s: number): V3 => [0.19 * s, 1.0, 0]
const ELBOW = (s: number): V3 => [0.22 * s, 0.79, 0]
const WRIST = (s: number): V3 => [0.235 * s, 0.6, 0]

/** Two toes per foot, side by side, so a single foot has a support area. */
const TOE_RADIUS = 0.035
const TOE_SPREAD = 0.03
const FOOT_ORIGIN = (s: number): V3 => [0.1 * s, TOE_RADIUS, 0.025]

/** Three fingers per hand, fanned front-to-back below the wrist. */
const FINGER_Z = [0.03, 0, -0.03] as const
const FINGER_BASE = (s: number, i: number): V3 => [0.235 * s, 0.585, FINGER_Z[i]]

// ---------------------------------------------------------------------------
// Parts, listed in assembly order (bottom-up)
// ---------------------------------------------------------------------------

const perSide = <T,>(fn: (side: (typeof SIDES)[number]) => T) => SIDES.map(fn)

function foot(tag: string, name: string, s: number): PartDef {
  const origin = FOOT_ORIGIN(s)
  const toe = (dx: number) =>
    capsule([origin[0] + dx, TOE_RADIUS, -0.09], [origin[0] + dx, TOE_RADIUS, 0.14], TOE_RADIUS, 0, origin)
  return {
    id: `foot_${tag}`,
    label: `${name} foot`,
    center: origin,
    shapes: [toe(-TOE_SPREAD), toe(TOE_SPREAD)],
    density: 1200,
    finish: 'graphite',
  }
}

export const PARTS: PartDef[] = [
  ...perSide(({ tag, name, s }) => foot(tag, name, s)),
  ...perSide(({ tag, name, s }) => bean(`shin_${tag}`, `${name} shin`, KNEE(s), ANKLE(s), 0.048)),
  ...perSide(({ tag, name, s }) => bean(`thigh_${tag}`, `${name} thigh`, HIP_PITCH(s), KNEE(s), 0.055)),
  ...perSide(({ tag, name, s }) =>
    ball(`hiproll_${tag}`, `${name} hip housing`, [0.1 * s, 0.535, 0], 0.04, { density: 4500, finish: 'graphite' }),
  ),
  ...perSide(({ tag, name, s }) =>
    ball(`hipyaw_${tag}`, `${name} hip rotator`, [0.1 * s, 0.578, 0], 0.032, { density: 6000, finish: 'graphite' }),
  ),
  bean('waist', 'Waist', [-0.1, 0.64, 0], [0.1, 0.64, 0], 0.07, { finish: 'graphite', gap: 0 }),
  bean('chest', 'Chest', [0, 0.74, 0], [0, 1.06, 0], 0.12, { gap: 0 }),
  ...perSide(({ tag, name, s }) => bean(`bicep_${tag}`, `${name} bicep`, SHOULDER(s), ELBOW(s), 0.045)),
  ...perSide(({ tag, name, s }) => bean(`forearm_${tag}`, `${name} forearm`, ELBOW(s), WRIST(s), 0.04)),
  ...perSide(({ tag, name, s }) =>
    FINGER_Z.map((_, i) =>
      bean(
        `finger${i + 1}_${tag}`,
        `${name} finger ${i + 1}`,
        FINGER_BASE(s, i),
        [FINGER_BASE(s, i)[0], 0.5, FINGER_BASE(s, i)[2]],
        0.016,
        { gap: 0.008, finish: 'graphite' },
      ),
    ),
  ).flat(),
  // Dense collar, for the same solver-stability reason as the hip housings.
  bean('neck', 'Neck', NECK, HEAD, 0.032, { finish: 'graphite', gap: 0.012, density: 7000 }),
  ball('head', 'Head', [0, 1.29, 0], 0.12, { density: 450 }),
]

export const PART_BY_ID = new Map(PARTS.map((p) => [p.id, p]))

// ---------------------------------------------------------------------------
// Joints: every connection is a single-axis revolute hinge with a motor
// ---------------------------------------------------------------------------

const X: V3 = [1, 0, 0]
const Y: V3 = [0, 1, 0]
const Z: V3 = [0, 0, 1]

function joint(
  id: string,
  label: string,
  parent: string,
  child: string,
  pivot: V3,
  axis: V3,
  limits: [number, number],
  strength = 1,
): JointDef {
  const local = (partId: string): V3 => {
    const part = PART_BY_ID.get(partId)
    if (!part) throw new Error(`Unknown part "${partId}" in joint "${id}"`)
    const c = part.center
    return [pivot[0] - c[0], pivot[1] - c[1], pivot[2] - c[2]]
  }
  return { id, label, parent, child, parentAnchor: local(parent), childAnchor: local(child), axis, limits, strength }
}

/** Mirror limits about zero for hinges whose axis doesn't flip with the side. */
const mirror = (s: number, [lo, hi]: [number, number]): [number, number] => (s > 0 ? [lo, hi] : [-hi, -lo])

export const JOINTS: JointDef[] = [
  joint('spine', 'Spine', 'waist', 'chest', SPINE, X, [-0.5, 0.6], 1.2),
  joint('neck', 'Neck', 'chest', 'neck', NECK, X, [-0.5, 0.5], 0.6),
  joint('head', 'Head yaw', 'neck', 'head', HEAD, Y, [-1.2, 1.2], 0.3),
  ...perSide(({ tag, name, s }) => [
    joint(`hip_yaw_${tag}`, `${name} hip yaw`, 'waist', `hipyaw_${tag}`, HIP_YAW(s), Y, [-0.6, 0.6], 1),
    joint(`hip_roll_${tag}`, `${name} hip roll`, `hipyaw_${tag}`, `hiproll_${tag}`, HIP_ROLL(s), Z, mirror(s, [-0.35, 0.6]), 1.4),
    joint(`hip_pitch_${tag}`, `${name} hip pitch`, `hiproll_${tag}`, `thigh_${tag}`, HIP_PITCH(s), X, [-1.7, 0.6], 1.4),
    joint(`knee_${tag}`, `${name} knee`, `thigh_${tag}`, `shin_${tag}`, KNEE(s), X, [-0.05, 2.4], 1.4),
    joint(`ankle_${tag}`, `${name} ankle`, `shin_${tag}`, `foot_${tag}`, ANKLE(s), X, [-0.8, 0.8], 1.2),
    joint(`shoulder_${tag}`, `${name} shoulder`, 'chest', `bicep_${tag}`, SHOULDER(s), Z, mirror(s, [-0.3, 2.6]), 0.6),
    joint(`elbow_${tag}`, `${name} elbow`, `bicep_${tag}`, `forearm_${tag}`, ELBOW(s), X, [-2.4, 0.05], 0.45),
    ...FINGER_Z.map((_, i) =>
      joint(
        `finger${i + 1}_${tag}`,
        `${name} finger ${i + 1}`,
        `forearm_${tag}`,
        `finger${i + 1}_${tag}`,
        FINGER_BASE(s, i),
        Z,
        mirror(s, [-1.4, 0.2]),
        0.08,
      ),
    ),
  ]).flat(),
]
