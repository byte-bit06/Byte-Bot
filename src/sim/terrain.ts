import type { V3 } from '../robot/skeleton'
import { createRng } from './random'

/** A static block on the ground: a slab, stepping stone or step. */
export interface TerrainBox {
  center: V3
  halfExtents: V3
  /** Rotation about the vertical axis, radians. */
  yaw: number
}

export interface Terrain {
  level: number
  seed: number
  /** Half-size of the flat ground plane, metres. */
  size: number
  boxes: TerrainBox[]
}

/**
 * Terrain curriculum. Level 0 is flat; each level scatters more, taller blocks
 * for Byte to walk over. The spawn area around the origin is always kept
 * clear so episodes start on flat ground.
 */
export const TERRAIN_LEVELS = [
  { name: 'flat', count: 0, maxHeight: 0 },
  { name: 'pebbles', count: 40, maxHeight: 0.015 },
  { name: 'rubble', count: 60, maxHeight: 0.03 },
  { name: 'steps', count: 70, maxHeight: 0.05 },
] as const

export const MAX_TERRAIN_LEVEL = TERRAIN_LEVELS.length - 1

export function makeTerrain(level: number, seed: number, radius = 4.5): Terrain {
  const spec = TERRAIN_LEVELS[Math.max(0, Math.min(MAX_TERRAIN_LEVEL, level))]
  const rng = createRng(seed)
  const boxes: TerrainBox[] = []
  while (boxes.length < spec.count) {
    const r = rng.range(0.6, radius)
    const a = rng.range(0, Math.PI * 2)
    const h = rng.range(spec.maxHeight * 0.4, spec.maxHeight) / 2
    boxes.push({
      center: [Math.cos(a) * r, h, Math.sin(a) * r],
      halfExtents: [rng.range(0.12, 0.4), h, rng.range(0.12, 0.4)],
      yaw: rng.range(0, Math.PI),
    })
  }
  return { level, seed, size: 30, boxes }
}
