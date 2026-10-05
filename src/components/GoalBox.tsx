import { useRef, type Ref } from 'react'
import { useFrame } from '@react-three/fiber'
import type { Mesh, MeshStandardMaterial } from 'three'

/** The green box Byte navigates to, with a pulsing ring on the floor. */
export function GoalBox({ materialRef }: { materialRef?: Ref<MeshStandardMaterial> }) {
  const ring = useRef<Mesh>(null)
  useFrame(({ clock }) => {
    if (!ring.current) return
    ring.current.scale.setScalar(1 + 0.15 * Math.sin(clock.elapsedTime * 3))
  })
  return (
    <>
      <mesh position-y={0.15} castShadow>
        <boxGeometry args={[0.3, 0.3, 0.3]} />
        <meshStandardMaterial
          ref={materialRef}
          color="#22c55e"
          emissive="#22c55e"
          emissiveIntensity={0.6}
          transparent
          opacity={0.75}
          roughness={0.3}
        />
      </mesh>
      <mesh ref={ring} rotation-x={-Math.PI / 2} position-y={0.004}>
        <ringGeometry args={[0.3, 0.35, 48]} />
        <meshBasicMaterial color="#22c55e" transparent opacity={0.6} toneMapped={false} />
      </mesh>
    </>
  )
}
