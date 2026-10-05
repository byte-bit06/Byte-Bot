import { useEffect, useMemo, useRef } from 'react'
import { useFrame } from '@react-three/fiber'
import { AdditiveBlending, Color, MeshBasicMaterial, MeshStandardMaterial, type Group, type Mesh } from 'three'
import type { PartDef, V3 } from '../robot/skeleton'

/**
 * Byte's visuals, shared by the story robot (React Three Rapier bodies) and
 * the policy-driven robot (headless simulator synced into the scene).
 */

export const COLORS = {
  shell: new Color('#f4f1ea'),
  graphite: new Color('#2b3038'),
  accent: new Color('#ff8a3d'),
  passive: new Color('#4a515c'),
  cyan: new Color('#22d3ee'),
  amber: new Color('#ff8a3d'),
  black: new Color('#000000'),
}

export const MATERIALS = {
  shell: new MeshStandardMaterial({ color: COLORS.shell, roughness: 0.36, metalness: 0.04 }),
  graphite: new MeshStandardMaterial({ color: COLORS.graphite, roughness: 0.5, metalness: 0.25 }),
  accent: new MeshStandardMaterial({ color: COLORS.accent, roughness: 0.4, metalness: 0.05 }),
  visor: new MeshStandardMaterial({ color: '#0b0d10', roughness: 0.15, metalness: 0.6 }),
  eye: new MeshBasicMaterial({ color: COLORS.cyan, toneMapped: false }),
}

/** The visible beans of one part, in the part's local frame. */
export function PartMeshes({ part }: { part: PartDef }) {
  return (
    <>
      {part.shapes.map((shape, i) =>
        shape.kind === 'capsule' ? (
          <mesh
            key={i}
            castShadow
            receiveShadow
            position={shape.offset}
            quaternion={shape.quaternion}
            material={MATERIALS[part.finish]}
          >
            <capsuleGeometry args={[shape.radius, shape.halfHeight * 2, 10, 24]} />
          </mesh>
        ) : (
          <mesh key={i} castShadow receiveShadow position={shape.offset} material={MATERIALS[part.finish]}>
            <sphereGeometry args={[shape.radius, 40, 28]} />
          </mesh>
        ),
      )}
      {part.id === 'head' && <Face />}
    </>
  )
}

/** Visor and blinking eyes, so Byte reads as a character. */
function Face() {
  const eyes = useRef<Group>(null)
  useFrame(({ clock }) => {
    if (!eyes.current) return
    eyes.current.scale.y = clock.elapsedTime % 4.2 < 0.12 ? 0.1 : 1
  })
  return (
    <group>
      <mesh position={[0, 0.012, 0.1]} rotation={[0, 0, Math.PI / 2]} scale={[1, 1, 0.5]} material={MATERIALS.visor}>
        <capsuleGeometry args={[0.05, 0.07, 8, 20]} />
      </mesh>
      <group ref={eyes} position={[0, 0.014, 0.124]}>
        {[-1, 1].map((s) => (
          <mesh key={s} position={[0.032 * s, 0, 0]} material={MATERIALS.eye}>
            <sphereGeometry args={[0.012, 16, 12]} />
          </mesh>
        ))}
      </group>
    </group>
  )
}

/** What a joint marker displays; mutated outside React every frame. */
export interface MarkerState {
  index: number
  /** Visibility (0 while its beans are still flying in). */
  reveal: number
  /** Cyan glow: an active, programmatically driven motor. */
  glow: number
  /** Amber flash: the joint just received a random command. */
  heat: number
}

const MARKER_RADIUS = 0.03

/** The visible pivot: graphite when passive, amber on random input, cyan when motorized. */
export function JointMarker({ state, position }: { state: MarkerState; position: V3 }) {
  const core = useRef<Mesh>(null)
  const halo = useRef<Mesh>(null)
  const coreMaterial = useMemo(
    () => new MeshStandardMaterial({ color: COLORS.passive, roughness: 0.3, metalness: 0.4, toneMapped: false }),
    [],
  )
  const haloMaterial = useMemo(
    () =>
      new MeshBasicMaterial({
        color: COLORS.cyan,
        transparent: true,
        opacity: 0,
        depthWrite: false,
        blending: AdditiveBlending,
        toneMapped: false,
      }),
    [],
  )
  useEffect(
    () => () => {
      coreMaterial.dispose()
      haloMaterial.dispose()
    },
    [coreMaterial, haloMaterial],
  )

  useFrame(({ clock }) => {
    if (!core.current || !halo.current) return
    const { glow, heat, reveal, index } = state
    const energy = Math.max(glow, heat)
    const pulse = 1 + 0.15 * glow * Math.sin(clock.elapsedTime * 4 + index * 0.7)
    core.current.scale.setScalar(Math.max(reveal * pulse, 1e-3))
    coreMaterial.color.copy(COLORS.passive).lerp(COLORS.cyan, glow).lerp(COLORS.amber, heat)
    coreMaterial.emissive.copy(COLORS.black).lerp(COLORS.cyan, glow).lerp(COLORS.amber, heat)
    coreMaterial.emissiveIntensity = 1.6
    halo.current.visible = energy > 0.01
    halo.current.scale.setScalar(reveal * (1.8 + 0.5 * pulse))
    haloMaterial.color.copy(COLORS.cyan).lerp(COLORS.amber, heat)
    haloMaterial.opacity = 0.28 * energy
  })

  return (
    <group position={position}>
      <mesh ref={core} material={coreMaterial}>
        <sphereGeometry args={[MARKER_RADIUS, 16, 12]} />
      </mesh>
      <mesh ref={halo} material={haloMaterial}>
        <sphereGeometry args={[MARKER_RADIUS, 16, 12]} />
      </mesh>
    </group>
  )
}
