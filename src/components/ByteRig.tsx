import { useMemo, type RefObject } from 'react'
import { useThree } from '@react-three/fiber'
import { Vector3, type Group } from 'three'
import type { OrbitControls as OrbitControlsImpl } from 'three-stdlib'
import { JOINTS, PARTS } from '../robot/skeleton'
import type { Terrain } from '../sim/terrain'
import { JointMarker, PartMeshes, type MarkerState } from './ByteParts'

/** Shared pieces for the policy-driven robots (live simulation and replay). */

export const WAIST = PARTS.findIndex((p) => p.id === 'waist')

const ACTIVE_MARKERS: MarkerState[] = JOINTS.map((_, index) => ({ index, reveal: 1, glow: 1, heat: 0 }))

/** Byte's parts as free-floating groups, posed every frame by the caller. */
export function ByteRig({ parts }: { parts: RefObject<(Group | null)[]> }) {
  return (
    <>
      {PARTS.map((part, i) => (
        <group
          key={part.id}
          ref={(g) => {
            parts.current[i] = g
          }}
        >
          <PartMeshes part={part} />
          {JOINTS.map((j, ji) =>
            j.parent === part.id ? (
              <JointMarker key={j.id} state={ACTIVE_MARKERS[ji]} position={j.parentAnchor} />
            ) : null,
          )}
        </group>
      ))}
    </>
  )
}

export function TerrainBlocks({ terrain }: { terrain: Terrain }) {
  return (
    <>
      {terrain.boxes.map((box, i) => (
        <mesh key={i} position={box.center} rotation-y={box.yaw} receiveShadow>
          <boxGeometry args={[box.halfExtents[0] * 2, box.halfExtents[1] * 2, box.halfExtents[2] * 2]} />
          <meshStandardMaterial color="#c9ced6" roughness={0.85} />
        </mesh>
      ))}
    </>
  )
}

/** Returns a per-frame function that eases the orbit camera after Byte's waist. */
export function useFollowCamera() {
  const controls = useThree((s) => s.controls) as OrbitControlsImpl | null
  const focus = useMemo(() => new Vector3(), [])
  return (waist: Group | null | undefined, dt: number) => {
    if (!controls || !waist) return
    focus.set(waist.position.x, 0.6, waist.position.z)
    const move = focus.sub(controls.target).multiplyScalar(1 - Math.exp(-dt * 2))
    controls.target.add(move)
    controls.object.position.add(move)
  }
}
