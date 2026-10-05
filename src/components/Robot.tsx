import { createRef, memo, useEffect, useMemo, useRef, useState, type RefObject } from 'react'
import { useThree } from '@react-three/fiber'
import {
  BallCollider,
  CapsuleCollider,
  RigidBody,
  useBeforePhysicsStep,
  useRapier,
  useRevoluteJoint,
  type RapierRigidBody,
} from '@react-three/rapier'
import type { RevoluteImpulseJoint } from '@dimforge/rapier3d-compat'
import { Euler, Quaternion, Vector3 } from 'three'
import { JOINTS, PARTS, type JointDef, type PartDef } from '../robot/skeleton'
import { DAMPING, FRICTION, MOTOR, ROBOT_COLLISION_GROUPS } from '../robot/physics'
import { clamp, clamp01, easeInOutCubic, easeOutBack, lerp, randRange, smoothstep } from '../robot/math'
import { JointMarker, PartMeshes, type MarkerState } from './ByteParts'

/**
 * How Byte's body is being driven.
 *
 * - `assemble`  every bean is kinematic and flies into its socket
 * - `chaos`     fully dynamic; joint motors receive random targets and gains
 * - `hold`      waist pinned in place, motors hold the rest pose
 * - `motorized` lifted onto a test stand; joints come online one by one and
 *               run a calibration sweep
 * - `stand`     lowered to the floor and released; motors hold a standing pose
 */
export type RobotMode = 'assemble' | 'chaos' | 'hold' | 'motorized' | 'stand'

export interface RobotProps {
  mode: RobotMode
  /** Fires once, when the `assemble` animation has finished. */
  onAssembled?: () => void
}

// ---------------------------------------------------------------------------
// Tuning
// ---------------------------------------------------------------------------

const ASSEMBLY = { stagger: 0.11, duration: 0.85, settle: 0.35 }
const STAND = { lowerSeconds: 1.4, uprightKp: 350, linearDamping: 2.5, angularDamping: 2 }
const TEST_STAND_LIFT = 0.22

// ---------------------------------------------------------------------------
// Per-joint runtime state, shared by the controller and the joint markers.
// Mutated every physics step, so it lives outside React state.
// ---------------------------------------------------------------------------

interface JointRuntime extends MarkerState {
  def: JointDef
  joint: RevoluteImpulseJoint | null
  chaos: { target: number; stiffness: number; nextAt: number }
}

const REST_POSITION = new Map(PARTS.map((p) => [p.id, new Vector3(...p.center)]))
const PART_INDEX = new Map(PARTS.map((p, i) => [p.id, i]))
const IDENTITY = new Quaternion()
const WORLD_UP = new Vector3(0, 1, 0)

const arrivalTime = (partIndex: number) => partIndex * ASSEMBLY.stagger + ASSEMBLY.duration
const ASSEMBLY_END = arrivalTime(PARTS.length - 1) + ASSEMBLY.settle

/** Where a bean starts before assembly: scattered in the air around the stage. */
function scatterPose() {
  const angle = randRange(0, Math.PI * 2)
  const radius = randRange(1.3, 2.4)
  return {
    position: new Vector3(Math.cos(angle) * radius, randRange(1.2, 2.8), Math.sin(angle) * radius - 0.4),
    rotation: new Quaternion().setFromEuler(new Euler(randRange(-3, 3), randRange(-3, 3), randRange(-3, 3))),
  }
}

// ---------------------------------------------------------------------------
// Robot
// ---------------------------------------------------------------------------

export function Robot({ mode, onAssembled }: RobotProps) {
  const { rapier } = useRapier()
  const pointer = useThree((s) => s.pointer)

  // The mode Byte is spawned in decides each body's initial type and pose.
  const [spawnMode] = useState(mode)
  const scatter = useMemo(() => PARTS.map(() => scatterPose()), [])

  const bodies = useMemo(() => new Map(PARTS.map((p) => [p.id, createRef<RapierRigidBody>()])), [])
  const runtime = useMemo<JointRuntime[]>(
    () =>
      JOINTS.map((def, index) => ({
        def,
        index,
        joint: null,
        reveal: spawnMode === 'assemble' ? 0 : 1,
        glow: spawnMode === 'stand' ? 1 : 0,
        heat: 0,
        chaos: { target: 0, stiffness: 0, nextAt: 0 },
      })),
    [spawnMode],
  )

  const sim = useRef({
    mode: spawnMode,
    t: 0,
    released: false,
    assembledReported: false,
    waistFrom: { position: new Vector3(), rotation: new Quaternion() },
    look: { yaw: 0, pitch: 0 },
  })

  // Dev-only handle for inspecting joints and bodies from the console.
  useEffect(() => {
    if (import.meta.env.DEV) (window as unknown as Record<string, unknown>).__byte = { runtime, bodies, sim }
  })
  const onAssembledRef = useRef(onAssembled)
  onAssembledRef.current = onAssembled

  // Scratch objects reused every step.
  const tmp = useMemo(() => ({ v: new Vector3(), q: new Quaternion(), up: new Vector3(), torque: new Vector3() }), [])

  // --- Mode transitions: switch rigid-body types and snapshot the waist ----
  useEffect(() => {
    const s = sim.current
    s.mode = mode
    s.t = 0
    s.released = false

    const waist = bodies.get('waist')?.current
    if (waist) {
      s.waistFrom.position.copy(waist.translation())
      s.waistFrom.rotation.copy(waist.rotation())
    }
    bodies.get('chest')?.current?.resetTorques(true)
    for (const ref of bodies.values()) {
      ref.current?.setLinearDamping(DAMPING.linear)
      ref.current?.setAngularDamping(DAMPING.angular)
    }

    const { Dynamic, KinematicPositionBased } = rapier.RigidBodyType
    for (const part of PARTS) {
      const body = bodies.get(part.id)?.current
      if (!body) continue
      const kinematic = mode === 'assemble' || (part.id === 'waist' && mode !== 'chaos')
      const type = kinematic ? KinematicPositionBased : Dynamic
      if (body.bodyType() !== type) body.setBodyType(type, true)
    }

    if (mode === 'chaos') {
      // A shove to get the flailing started.
      bodies.get('chest')?.current?.applyTorqueImpulse(
        { x: randRange(-4, 4), y: randRange(-1, 1), z: randRange(-4, 4) },
        true,
      )
    }
  }, [mode, bodies, rapier])

  // --- Controllers --------------------------------------------------------

  const body = (id: string) => bodies.get(id)?.current ?? null

  const setDamping = (linear: number, angular: number) => {
    for (const ref of bodies.values()) {
      ref.current?.setLinearDamping(linear)
      ref.current?.setAngularDamping(angular)
    }
  }

  const drive = (rt: JointRuntime, target: number, gain = 1, dampingGain = gain) => {
    const [lo, hi] = rt.def.limits
    rt.joint?.configureMotorPosition(
      clamp(target, lo, hi),
      MOTOR.stiffness * rt.def.strength * gain,
      MOTOR.damping * rt.def.strength * dampingGain,
    )
  }

  /** Glide the kinematic waist from where it was when the mode began. */
  const glideWaist = (to: Vector3, seconds: number) => {
    const waist = body('waist')
    if (!waist) return
    const s = sim.current
    const k = smoothstep(s.t / seconds)
    waist.setNextKinematicTranslation(tmp.v.lerpVectors(s.waistFrom.position, to, k))
    waist.setNextKinematicRotation(tmp.q.slerpQuaternions(s.waistFrom.rotation, IDENTITY, k))
  }

  const stepAssemble = (t: number) => {
    PARTS.forEach((part, i) => {
      const b = body(part.id)
      if (!b) return
      const k = (t - i * ASSEMBLY.stagger) / ASSEMBLY.duration
      const from = scatter[i]
      b.setNextKinematicTranslation(tmp.v.lerpVectors(from.position, REST_POSITION.get(part.id)!, easeOutBack(k)))
      b.setNextKinematicRotation(tmp.q.slerpQuaternions(from.rotation, IDENTITY, easeInOutCubic(k)))
    })
    for (const rt of runtime) {
      const landed = Math.max(arrivalTime(PART_INDEX.get(rt.def.parent)!), arrivalTime(PART_INDEX.get(rt.def.child)!))
      rt.reveal = clamp01((t - landed) / 0.25)
    }
    const s = sim.current
    if (t >= ASSEMBLY_END && !s.assembledReported) {
      s.assembledReported = true
      onAssembledRef.current?.()
    }
  }

  const stepChaos = (t: number, dt: number) => {
    // Full-strength flailing at first, then weaker twitching once Byte is down.
    const intensity = t < 3 ? 1 : Math.max(0.3, 1 - (t - 3) * 0.25)
    for (const rt of runtime) {
      const c = rt.chaos
      if (t >= c.nextAt) {
        c.target = lerp(rt.def.limits[0], rt.def.limits[1], Math.random())
        c.stiffness = randRange(0.25, 1.6)
        c.nextAt = t + randRange(0.12, 0.5)
        rt.heat = 1
      }
      rt.heat = Math.max(0, rt.heat - dt * 3)
      rt.glow = 0
      drive(rt, c.target, c.stiffness * intensity, 0.12)
    }
  }

  const stepHold = () => {
    glideWaist(REST_POSITION.get('waist')!, 0.8)
    for (const rt of runtime) {
      rt.glow = 0
      rt.heat = 0
      drive(rt, 0)
    }
  }

  const stepMotorized = (t: number) => {
    glideWaist(tmp.up.copy(REST_POSITION.get('waist')!).setY(REST_POSITION.get('waist')!.y + TEST_STAND_LIFT), 1.2)
    for (const rt of runtime) {
      // Limp at first, then each joint is swapped for a motor in sequence.
      const online = clamp01((t - 0.9 - rt.index * 0.14) / 0.3)
      rt.glow = online
      rt.heat = 0
      const [lo, hi] = rt.def.limits
      const amp = rt.def.strength < 0.2 ? 0.7 : 0.35
      const center = clamp(0, lo + amp, hi - amp)
      const sweep = center + amp * Math.sin(1.8 * t + rt.index * 0.55)
      drive(rt, online > 0 ? sweep : 0, online, Math.max(online, 0.03))
    }
  }

  const stepStand = (t: number, dt: number) => {
    const s = sim.current
    const waist = body('waist')
    if (!waist) return

    if (t < STAND.lowerSeconds) {
      glideWaist(REST_POSITION.get('waist')!, STAND.lowerSeconds)
    } else if (!s.released) {
      waist.setBodyType(rapier.RigidBodyType.Dynamic, true)
      setDamping(STAND.linearDamping, STAND.angularDamping)
      s.released = true
    }

    // Idle behaviour while waiting for a policy: breathe and watch the cursor.
    const ease = 1 - Math.exp(-dt * 4)
    s.look.yaw = lerp(s.look.yaw, 0.55 + pointer.x * 0.7, ease)
    s.look.pitch = lerp(s.look.pitch, -pointer.y * 0.3, ease)

    for (const rt of runtime) {
      rt.glow = 1
      rt.heat = 0
      let target = 0
      switch (rt.def.id) {
        case 'spine':
          target = 0.03 * Math.sin(t * 1.3)
          break
        case 'head':
          target = s.look.yaw
          break
        case 'neck':
          target = s.look.pitch
          break
      }
      drive(rt, target)
    }

    // Placeholder balance assist until the RL policy exists. A stiff body on
    // small feet is only marginally stable, so a proportional torque keeps the
    // chest upright; the damping comes from the bodies' (implicit) damping set
    // on release. Remove this once a trained policy is in charge.
    const chest = body('chest')
    if (s.released && chest) {
      const up = tmp.up.copy(WORLD_UP).applyQuaternion(tmp.q.copy(chest.rotation()))
      chest.resetTorques(false)
      chest.addTorque(tmp.torque.crossVectors(up, WORLD_UP).multiplyScalar(STAND.uprightKp), true)
    }
  }

  useBeforePhysicsStep((world) => {
    const s = sim.current
    const dt = world.timestep
    s.t += dt
    switch (s.mode) {
      case 'assemble':
        return stepAssemble(s.t)
      case 'chaos':
        return stepChaos(s.t, dt)
      case 'hold':
        return stepHold()
      case 'motorized':
        return stepMotorized(s.t)
      case 'stand':
        return stepStand(s.t, dt)
    }
  })

  // --- Scene graph --------------------------------------------------------

  return (
    <group name="byte">
      {PARTS.map((part, i) => {
        const assembling = spawnMode === 'assemble'
        return (
          <Bean
            key={part.id}
            part={part}
            bodyRef={bodies.get(part.id)!}
            kinematic={assembling || (part.id === 'waist' && spawnMode !== 'chaos')}
            position={assembling ? scatter[i].position : REST_POSITION.get(part.id)!}
            rotation={assembling ? scatter[i].rotation : IDENTITY}
            markers={runtime.filter((rt) => rt.def.parent === part.id)}
          />
        )
      })}
      {runtime.map((rt) => (
        <MotorJoint
          key={rt.def.id}
          rt={rt}
          parent={bodies.get(rt.def.parent)!}
          child={bodies.get(rt.def.child)!}
          initialGain={spawnMode === 'chaos' ? 0 : 1}
        />
      ))}
    </group>
  )
}

// ---------------------------------------------------------------------------
// A single part: one rigid body with its bean collider(s) and meshes
// ---------------------------------------------------------------------------

interface BeanProps {
  part: PartDef
  bodyRef: RefObject<RapierRigidBody | null>
  kinematic: boolean
  position: Vector3
  rotation: Quaternion
  markers: JointRuntime[]
}

const Bean = memo(function Bean({ part, bodyRef, kinematic, position, rotation, markers }: BeanProps) {
  // Spawn-time transform only; afterwards the physics engine owns the pose.
  const [spawn] = useState(() => ({
    position: position.toArray(),
    rotation: new Euler().setFromQuaternion(rotation).toArray() as [number, number, number],
  }))

  return (
    <RigidBody
      ref={bodyRef as RefObject<RapierRigidBody>}
      name={part.id}
      type={kinematic ? 'kinematicPosition' : 'dynamic'}
      position={spawn.position}
      rotation={spawn.rotation}
      colliders={false}
      canSleep={false}
      linearDamping={DAMPING.linear}
      angularDamping={DAMPING.angular}
    >
      {part.shapes.map((shape, i) =>
        shape.kind === 'capsule' ? (
          <CapsuleCollider
            key={i}
              args={[shape.halfHeight, shape.radius]}
              position={shape.offset}
              quaternion={shape.quaternion}
              density={part.density}
              friction={FRICTION.capsule}
              restitution={0}
              collisionGroups={ROBOT_COLLISION_GROUPS}
            />
        ) : (
          <BallCollider
            key={i}
              args={[shape.radius]}
              position={shape.offset}
              density={part.density}
              friction={FRICTION.ball}
              restitution={0}
              collisionGroups={ROBOT_COLLISION_GROUPS}
            />
        ),
      )}
      <PartMeshes part={part} />
      {markers.map((rt) => (
        <JointMarker key={rt.def.id} state={rt} position={rt.def.parentAnchor} />
      ))}
    </RigidBody>
  )
})

// ---------------------------------------------------------------------------
// Joints
// ---------------------------------------------------------------------------

interface MotorJointProps {
  rt: JointRuntime
  parent: RefObject<RapierRigidBody | null>
  child: RefObject<RapierRigidBody | null>
  initialGain: number
}

/**
 * A revolute (hinge) impulse joint with its motor enabled. Unlike a ragdoll's
 * passive joints, this one takes a target angle plus PD gains every step.
 */
function MotorJoint({ rt, parent, child, initialGain }: MotorJointProps) {
  const { rapier } = useRapier()
  const { def } = rt
  const jointRef = useRevoluteJoint(
    parent as RefObject<RapierRigidBody>,
    child as RefObject<RapierRigidBody>,
    [def.parentAnchor, def.childAnchor, def.axis, def.limits],
  )

  useEffect(() => {
    const joint = jointRef.current
    if (!joint) return
    joint.setContactsEnabled(false)
    joint.configureMotorModel(rapier.MotorModel.ForceBased)
    joint.configureMotorPosition(
      0,
      MOTOR.stiffness * def.strength * initialGain,
      MOTOR.damping * def.strength * initialGain,
    )
    rt.joint = joint
    return () => {
      rt.joint = null
    }
  }, [jointRef, rt, def, rapier, initialGain])

  return null
}
