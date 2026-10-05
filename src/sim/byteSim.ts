import type RAPIER_NS from '@dimforge/rapier3d-compat'
import type { Collider, RevoluteImpulseJoint, RigidBody, World } from '@dimforge/rapier3d-compat'
import { JOINTS, PARTS, type JointDef } from '../robot/skeleton'
import { DAMPING, FRICTION, MOTOR, ROBOT_COLLISION_GROUPS, WORLD } from '../robot/physics'
import type { Terrain } from './terrain'

export type Rapier = typeof RAPIER_NS

type Vec = { x: number; y: number; z: number }
type Quat = { x: number; y: number; z: number; w: number }

// Small quaternion helpers on plain objects (no three.js in the hot loop).
const qConj = (q: Quat): Quat => ({ x: -q.x, y: -q.y, z: -q.z, w: q.w })
const qMul = (a: Quat, b: Quat): Quat => ({
  w: a.w * b.w - a.x * b.x - a.y * b.y - a.z * b.z,
  x: a.w * b.x + a.x * b.w + a.y * b.z - a.z * b.y,
  y: a.w * b.y - a.x * b.z + a.y * b.w + a.z * b.x,
  z: a.w * b.z + a.x * b.y - a.y * b.x + a.z * b.w,
})
export const qRotate = (q: Quat, v: Vec): Vec => {
  // v' = q v q*, expanded.
  const ix = q.w * v.x + q.y * v.z - q.z * v.y
  const iy = q.w * v.y + q.z * v.x - q.x * v.z
  const iz = q.w * v.z + q.x * v.y - q.y * v.x
  const iw = -q.x * v.x - q.y * v.y - q.z * v.z
  return {
    x: ix * q.w + iw * -q.x + iy * -q.z - iz * -q.y,
    y: iy * q.w + iw * -q.y + iz * -q.x - ix * -q.z,
    z: iz * q.w + iw * -q.z + ix * -q.y - iy * -q.x,
  }
}
export const qYaw = (yaw: number): Quat => ({ x: 0, y: Math.sin(yaw / 2), z: 0, w: Math.cos(yaw / 2) })

/** Ground contact flags for the parts the RL environment cares about. */
export interface ContactState {
  footL: boolean
  footR: boolean
  /** Any part other than a foot (or shin tip) is touching the ground: a fall. */
  body: boolean
}

/**
 * Byte in a bare Rapier world: no React, no three.js scene graph. Built from
 * the same skeleton and physics constants as the website's Robot component,
 * so a policy trained here runs unchanged on the site.
 */
export class ByteSim {
  readonly world: World
  readonly bodies = new Map<string, RigidBody>()
  readonly joints = new Map<string, RevoluteImpulseJoint>()
  readonly jointDefs = new Map<string, JointDef>(JOINTS.map((j) => [j.id, j]))
  private readonly groundColliders: Collider[] = []
  private readonly footColliders = { L: [] as Collider[], R: [] as Collider[] }
  private readonly fallColliders: Collider[] = []
  private readonly R: Rapier

  constructor(R: Rapier, terrain: Terrain) {
    this.R = R
    this.world = new R.World({ x: 0, y: WORLD.gravity, z: 0 })
    this.world.timestep = WORLD.timestep
    this.world.integrationParameters.numSolverIterations = WORLD.solverIterations

    this.buildTerrain(terrain)
    this.buildRobot()
  }

  private buildTerrain(terrain: Terrain) {
    const { R, world } = this
    const ground = world.createRigidBody(R.RigidBodyDesc.fixed())
    this.groundColliders.push(
      world.createCollider(
        R.ColliderDesc.cuboid(terrain.size, 0.5, terrain.size).setTranslation(0, -0.5, 0).setFriction(FRICTION.ground),
        ground,
      ),
    )
    for (const box of terrain.boxes) {
      const desc = R.ColliderDesc.cuboid(...box.halfExtents)
        .setTranslation(...box.center)
        .setRotation(qYaw(box.yaw))
        .setFriction(FRICTION.ground)
      this.groundColliders.push(world.createCollider(desc, ground))
    }
  }

  private buildRobot() {
    const { R, world } = this
    for (const part of PARTS) {
      const body = world.createRigidBody(
        R.RigidBodyDesc.dynamic()
          .setTranslation(...part.center)
          .setLinearDamping(DAMPING.linear)
          .setAngularDamping(DAMPING.angular)
          .setCanSleep(false),
      )
      for (const shape of part.shapes) {
        const desc =
          shape.kind === 'capsule'
            ? R.ColliderDesc.capsule(shape.halfHeight, shape.radius).setRotation({
                x: shape.quaternion[0],
                y: shape.quaternion[1],
                z: shape.quaternion[2],
                w: shape.quaternion[3],
              })
            : R.ColliderDesc.ball(shape.radius)
        desc
          .setTranslation(...shape.offset)
          .setDensity(part.density)
          .setFriction(shape.kind === 'capsule' ? FRICTION.capsule : FRICTION.ball)
          .setRestitution(0)
          .setCollisionGroups(ROBOT_COLLISION_GROUPS)
        const collider = world.createCollider(desc, body)
        if (part.id === 'foot_L') this.footColliders.L.push(collider)
        else if (part.id === 'foot_R') this.footColliders.R.push(collider)
        else if (!part.id.startsWith('shin_')) this.fallColliders.push(collider)
      }
      this.bodies.set(part.id, body)
    }

    for (const def of JOINTS) {
      const v = (a: readonly number[]) => ({ x: a[0], y: a[1], z: a[2] })
      const data = R.JointData.revolute(v(def.parentAnchor), v(def.childAnchor), v(def.axis))
      data.limitsEnabled = true
      data.limits = def.limits
      const joint = world.createImpulseJoint(
        data,
        this.bodies.get(def.parent)!,
        this.bodies.get(def.child)!,
        true,
      ) as RevoluteImpulseJoint
      joint.setContactsEnabled(false)
      joint.configureMotorModel(R.MotorModel.ForceBased)
      joint.configureMotorPosition(0, MOTOR.stiffness * def.strength, MOTOR.damping * def.strength)
      this.joints.set(def.id, joint)
    }
  }

  /** Teleport Byte to its rest pose at (x, z) facing `yaw`, at rest. */
  reset(x = 0, z = 0, yaw = 0, lift = 0.005) {
    const q = qYaw(yaw)
    for (const part of PARTS) {
      const body = this.bodies.get(part.id)!
      const p = qRotate(q, { x: part.center[0], y: part.center[1], z: part.center[2] })
      body.setTranslation({ x: p.x + x, y: p.y + lift, z: p.z + z }, true)
      body.setRotation(q, true)
      body.setLinvel({ x: 0, y: 0, z: 0 }, true)
      body.setAngvel({ x: 0, y: 0, z: 0 }, true)
      body.resetForces(true)
      body.resetTorques(true)
    }
    for (const def of JOINTS) this.drive(def.id, 0)
  }

  /** PD position target for one joint; `gain` scales the joint's nominal stiffness. */
  drive(jointId: string, target: number, gain = 1) {
    const def = this.jointDefs.get(jointId)!
    const [lo, hi] = def.limits
    this.joints
      .get(jointId)!
      .configureMotorPosition(
        Math.min(hi, Math.max(lo, target)),
        MOTOR.stiffness * def.strength * gain,
        MOTOR.damping * def.strength * gain,
      )
  }

  /** Relative hinge angle (child w.r.t. parent) about the joint axis, in radians. */
  jointAngle(jointId: string): number {
    const def = this.jointDefs.get(jointId)!
    const rel = qMul(qConj(this.bodies.get(def.parent)!.rotation()), this.bodies.get(def.child)!.rotation())
    const s = rel.x * def.axis[0] + rel.y * def.axis[1] + rel.z * def.axis[2]
    let angle = 2 * Math.atan2(s, rel.w)
    if (angle > Math.PI) angle -= 2 * Math.PI
    if (angle < -Math.PI) angle += 2 * Math.PI
    return angle
  }

  /** Relative hinge angular velocity about the joint axis (rad/s). */
  jointVelocity(jointId: string): number {
    const def = this.jointDefs.get(jointId)!
    const parent = this.bodies.get(def.parent)!
    const axis = qRotate(parent.rotation(), { x: def.axis[0], y: def.axis[1], z: def.axis[2] })
    const wp = parent.angvel()
    const wc = this.bodies.get(def.child)!.angvel()
    return (wc.x - wp.x) * axis.x + (wc.y - wp.y) * axis.y + (wc.z - wp.z) * axis.z
  }

  private touchingGround(colliders: Collider[]): boolean {
    for (const c of colliders) {
      for (const g of this.groundColliders) {
        let touching = false
        this.world.contactPair(c, g, (manifold) => {
          if (manifold.numSolverContacts() > 0) touching = true
        })
        if (touching) return true
      }
    }
    return false
  }

  contacts(): ContactState {
    return {
      footL: this.touchingGround(this.footColliders.L),
      footR: this.touchingGround(this.footColliders.R),
      body: this.touchingGround(this.fallColliders),
    }
  }

  step() {
    this.world.step()
  }

  free() {
    this.world.free()
  }
}
