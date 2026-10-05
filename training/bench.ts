/**
 * Sanity checks for the headless simulator:
 *   1. joint-angle sign convention matches Rapier's motor targets
 *   2. Byte's mass and whether the rest pose stands unassisted
 *   3. raw physics throughput (steps/second) on one core
 *
 *   npx tsx training/bench.ts
 */
import RAPIER from '@dimforge/rapier3d-compat'
import { ByteSim } from '../src/sim/byteSim'
import { makeTerrain } from '../src/sim/terrain'
import { JOINTS } from '../src/robot/skeleton'

await RAPIER.init()

// 1. Sign convention: pin the waist in the air, drive every joint to +0.3 rad.
{
  const sim = new ByteSim(RAPIER, makeTerrain(0, 1))
  sim.reset(0, 0, 0, 0.6)
  sim.bodies.get('waist')!.setBodyType(RAPIER.RigidBodyType.Fixed, true)
  const target = (lo: number, hi: number) => Math.min(hi, Math.max(lo, 0.3))
  for (const j of JOINTS) sim.drive(j.id, target(...j.limits))
  for (let i = 0; i < 240; i++) sim.step()
  const bad = JOINTS.filter((j) => Math.abs(sim.jointAngle(j.id) - target(...j.limits)) > 0.08).map(
    (j) => `${j.id}: wanted ${target(...j.limits).toFixed(2)} got ${sim.jointAngle(j.id).toFixed(2)}`,
  )
  console.log(bad.length ? `SIGN/TRACKING MISMATCH\n  ${bad.join('\n  ')}` : 'joint angles track motor targets ✓')
  sim.free()
}

// 2. Mass and passive standing.
{
  const sim = new ByteSim(RAPIER, makeTerrain(0, 1))
  sim.reset()
  let mass = 0
  for (const b of sim.bodies.values()) mass += b.mass()
  console.log(`total mass ${mass.toFixed(1)} kg over ${sim.bodies.size} bodies, ${sim.joints.size} joints`)
  for (let i = 0; i < 120 * 5; i++) sim.step()
  const waist = sim.bodies.get('waist')!.translation()
  console.log(`after 5 s holding rest pose: waist y=${waist.y.toFixed(3)} z=${waist.z.toFixed(3)} contacts`, sim.contacts())
  sim.free()
}

// 3. Throughput.
{
  const sim = new ByteSim(RAPIER, makeTerrain(2, 7))
  sim.reset()
  const n = 6000
  const t0 = performance.now()
  for (let i = 0; i < n; i++) {
    if (i % 3 === 0) {
      for (const j of JOINTS) sim.drive(j.id, Math.sin(i * 0.01 + j.id.length) * 0.2)
      sim.contacts()
    }
    sim.step()
    if (i % 600 === 0) sim.reset()
  }
  const ms = performance.now() - t0
  console.log(`${((n / ms) * 1000).toFixed(0)} physics steps/s on one core (${(ms / n * 1000).toFixed(1)} µs/step)`)
  sim.free()
}
