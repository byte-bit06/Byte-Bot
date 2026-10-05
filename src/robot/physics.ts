/**
 * Physics constants shared by the website (React Three Fiber) and the
 * headless training simulator. Keeping them in one place guarantees the policy
 * is trained on exactly the physics it will be deployed on.
 */

export const WORLD = {
  gravity: -9.81,
  /** Physics step. Policies act every `CONTROL_DECIMATION` steps. */
  timestep: 1 / 120,
  solverIterations: 10,
} as const

/**
 * Byte's colliders belong to group 1 and only collide with group 0 (the
 * ground), never with each other. Same bit layout as Rapier's
 * `interactionGroups(1, [0])`: memberships in the high 16 bits.
 */
export const ROBOT_COLLISION_GROUPS = ((1 << 1) << 16) | (1 << 0)

/** Base PD gains for a force-based motor (N·m/rad, N·m·s/rad); scaled by each joint's `strength`. */
export const MOTOR = { stiffness: 900, damping: 45 } as const

export const DAMPING = { linear: 0.05, angular: 0.6 } as const

export const FRICTION = { capsule: 1.4, ball: 1, ground: 1.2 } as const
