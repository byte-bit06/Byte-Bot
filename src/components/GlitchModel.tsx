import { useEffect, useMemo, useRef } from 'react'
import { useFrame } from '@react-three/fiber'
import {
  CapsuleGeometry,
  MeshBasicMaterial,
  Quaternion,
  ShaderMaterial,
  SphereGeometry,
  Vector3,
  type BufferGeometry,
  type Group,
} from 'three'
import { PARTS } from '../robot/skeleton'
import { randRange } from '../robot/math'

/**
 * Stand-in for the "26.9 MB rigged character": the same silhouette as Byte,
 * but dense, spiky geometry whose pieces get flung around as if the solver
 * had diverged. Purely visual; nothing here touches the physics world.
 */

const vertexShader = /* glsl */ `
  uniform float uTime;
  varying vec3 vNormal;
  varying float vSpike;

  float hash(vec3 p) {
    p = fract(p * 0.3183099 + 0.1);
    p *= 17.0;
    return fract(p.x * p.y * p.z * (p.x + p.y + p.z));
  }

  void main() {
    // Stepped time makes the distortion jump instead of flow.
    float t = floor(uTime * 14.0);
    float h = hash(floor(position * 22.0) + t);
    float spike = step(0.8, h) * (h - 0.8) * 5.0;
    float band = step(0.88, hash(vec3(floor(position.y * 45.0), t, 3.0)));

    vec3 p = position + normal * spike * 0.09;
    p.x += band * (hash(vec3(t, position.y, 7.0)) - 0.5) * 0.12;

    vSpike = spike;
    vNormal = normalize(normalMatrix * normal);
    gl_Position = projectionMatrix * modelViewMatrix * vec4(p, 1.0);
  }
`

const fragmentShader = /* glsl */ `
  uniform float uTime;
  varying vec3 vNormal;
  varying float vSpike;

  void main() {
    float light = 0.35 + 0.65 * max(dot(vNormal, normalize(vec3(0.4, 0.8, 0.5))), 0.0);
    vec3 base = mix(vec3(0.82, 0.8, 0.86), vec3(1.0, 0.17, 0.84), vSpike);
    vec3 color = base * light;
    // Scanlines and the occasional dropped row.
    if (mod(gl_FragCoord.y, 3.0) < 1.0) color *= 0.55;
    if (fract(sin(floor(gl_FragCoord.y * 0.25) + floor(uTime * 20.0)) * 43758.5) > 0.97) color = vec3(0.13, 0.83, 0.93);
    gl_FragColor = vec4(color, 1.0);
  }
`

const REST = PARTS.map((p) => new Vector3(...p.center))
const BODY_CENTER = new Vector3(0, 0.7, 0)
const ONE = new Vector3(1, 1, 1)

export function GlitchModel() {
  const root = useRef<Group>(null)
  const parts = useRef<(Group | null)[]>([])
  const burst = useRef(-1)

  const material = useMemo(
    () => new ShaderMaterial({ vertexShader, fragmentShader, uniforms: { uTime: { value: 0 } } }),
    [],
  )
  const wire = useMemo(
    () => new MeshBasicMaterial({ color: '#ff2bd6', wireframe: true, transparent: true, opacity: 0.35 }),
    [],
  )
  // Deliberately heavy tessellation: this is the "high-poly" model.
  const geometries = useMemo<BufferGeometry[][]>(
    () =>
      PARTS.map(({ shapes }) =>
        shapes.map((shape) =>
          shape.kind === 'capsule'
            ? new CapsuleGeometry(shape.radius, shape.halfHeight * 2, 24, 64, 12)
            : new SphereGeometry(shape.radius, 96, 64),
        ),
      ),
    [],
  )

  useEffect(
    () => () => {
      geometries.flat().forEach((g) => g.dispose())
      material.dispose()
      wire.dispose()
    },
    [geometries, material, wire],
  )

  useFrame(({ clock }) => {
    const t = clock.elapsedTime
    material.uniforms.uTime.value = t
    if (!root.current) return

    root.current.visible = Math.random() > 0.07
    root.current.position.set(randRange(-0.03, 0.03), randRange(-0.01, 0.02), randRange(-0.03, 0.03))

    // Every ~0.9 s the whole model "explodes" outward for a frame or two.
    const tick = Math.floor(t / 0.9)
    const exploding = tick !== burst.current
    burst.current = tick

    parts.current.forEach((g, i) => {
      if (!g) return
      const rest = REST[i]
      if (exploding) {
        g.position.copy(rest).sub(BODY_CENTER).multiplyScalar(randRange(1.4, 2.2)).add(BODY_CENTER)
      } else if (Math.random() < 0.08) {
        g.position.copy(rest).add(new Vector3(randRange(-1, 1), randRange(-1, 1), randRange(-1, 1)).multiplyScalar(0.3))
        const axis = Math.floor(Math.random() * 3)
        g.scale.set(1, 1, 1).setComponent(axis, randRange(1.5, 3.5))
        g.quaternion.setFromAxisAngle(new Vector3(Math.random(), Math.random(), Math.random()).normalize(), randRange(-1, 1))
      } else {
        g.position.lerp(rest, 0.3)
        g.scale.lerp(ONE, 0.25)
        g.quaternion.slerp(new Quaternion(), 0.25)
      }
    })
  })

  return (
    <group ref={root}>
      {PARTS.map((part, i) => (
        <group
          key={part.id}
          ref={(g) => {
            parts.current[i] = g
          }}
          position={part.center}
        >
          {part.shapes.map((shape, j) => {
            const q = shape.kind === 'capsule' ? shape.quaternion : undefined
            return (
              <group key={j} position={shape.offset} quaternion={q}>
                <mesh geometry={geometries[i][j]} material={material} castShadow />
                <mesh geometry={geometries[i][j]} material={wire} scale={1.04} position={[0.012, 0, 0]} />
              </group>
            )
          })}
        </group>
      ))}
    </group>
  )
}
