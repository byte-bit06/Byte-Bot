import { useEffect, useMemo } from 'react'
import { useFrame, useThree } from '@react-three/fiber'
import { Environment, Grid, Lightformer, OrbitControls } from '@react-three/drei'
import { CuboidCollider, Physics, RigidBody } from '@react-three/rapier'
import { Vector3 } from 'three'
import type { OrbitControls as OrbitControlsImpl } from 'three-stdlib'
import { WORLD } from '../robot/physics'
import { useLab } from '../story/lab'
import { useStory } from '../story/store'
import { stageDef } from '../story/stages'
import { GlitchModel } from './GlitchModel'
import { GoalBox } from './GoalBox'
import { PolicyRobot } from './PolicyRobot'
import { ReplayRobot } from './ReplayRobot'
import { Robot } from './Robot'

const BACKGROUND = '#eceef1'

export function Scene() {
  const stage = useStory((s) => s.stage)
  const glitching = useStory((s) => s.glitching)
  const robotKey = useStory((s) => s.robotKey)
  const completeBeat = useStory((s) => s.completeBeat)
  const policy = useLab((s) => s.policy)
  const terrainLevel = useLab((s) => s.terrainLevel)
  const liveMode = useLab((s) => s.liveMode)
  const manifest = useLab((s) => s.manifest)
  const checkpoint = useLab((s) => s.checkpoint)
  const mode = stageDef(stage).robotMode
  const liveActive = stage === 7 && liveMode && policy !== null

  return (
    <>
      <color attach="background" args={[BACKGROUND]} />
      <fog attach="fog" args={[BACKGROUND, 9, 24]} />

      <StudioLights />

      <Physics gravity={[0, WORLD.gravity, 0]} timeStep={WORLD.timestep} numSolverIterations={WORLD.solverIterations}>
        <Floor />
        {!glitching && mode !== 'policy' && <Robot key={robotKey} mode={mode} onAssembled={completeBeat} />}
      </Physics>

      {glitching && <GlitchModel />}

      {/* Stage 5 previews the navigation goal next to the standing robot. */}
      {stage === 5 && (
        <group position={[0.7, 0, 1.4]}>
          <GoalBox />
        </group>
      )}

      {/* Live network + physics (stage 7 only). Each policy or terrain change starts afresh. */}
      {liveActive && <PolicyRobot key={`${policy.meta.iteration}-${terrainLevel}`} policyFile={policy} terrainLevel={terrainLevel} />}
      {/* Otherwise, playback of recorded rollouts; it also covers the moment while
          the live network loads, so switching modes never shows an empty scene. */}
      {mode === 'policy' && !liveActive && manifest && (
        <ReplayRobot
          file={
            stage === 7
              ? (manifest.navigation.find((n) => n.terrainLevel === terrainLevel) ?? manifest.navigation[0]).file
              : manifest.checkpoints[checkpoint].clip
          }
        />
      )}
      {stage === 7 && <GoalClickTarget />}
      {mode !== 'policy' && <CameraHome />}
      <ViewFraming />

      <OrbitControls
        makeDefault
        target={[0, 0.7, 0]}
        enablePan={false}
        enableDamping
        minDistance={1.8}
        maxDistance={7}
        maxPolarAngle={Math.PI / 2 - 0.06}
      />
    </>
  )
}

/**
 * Shift the rendered view so Byte isn't hidden behind the narration panel:
 * right of the panel on wide screens, above it on narrow ones.
 */
function ViewFraming() {
  const camera = useThree((s) => s.camera)
  const size = useThree((s) => s.size)
  useEffect(() => {
    if (!('setViewOffset' in camera)) return
    const { width: w, height: h } = size
    if (w >= 768) camera.setViewOffset(w, h, -w * 0.17, 0, w, h)
    else camera.setViewOffset(w, h, 0, h * 0.2, w, h)
    camera.updateProjectionMatrix()
    return () => {
      camera.clearViewOffset()
      camera.updateProjectionMatrix()
    }
  }, [camera, size])
  return null
}

/**
 * Invisible floor plane for stage 7: a click places the goal (and switches to
 * the live network). Drags that orbit the camera are ignored.
 */
function GoalClickTarget() {
  const placeGoal = useLab((s) => s.placeGoal)
  useEffect(() => () => void (document.body.style.cursor = ''), [])
  return (
    <mesh
      rotation-x={-Math.PI / 2}
      position-y={0.002}
      visible={false}
      onClick={(e) => {
        if (e.delta > 6) return
        e.stopPropagation()
        placeGoal(e.point.x, e.point.z)
      }}
      onPointerOver={() => (document.body.style.cursor = 'crosshair')}
      onPointerOut={() => (document.body.style.cursor = '')}
    >
      <planeGeometry args={[60, 60]} />
    </mesh>
  )
}

/** Ease the orbit target back to Byte's start after following a walking policy. */
function CameraHome() {
  const controls = useThree((s) => s.controls) as OrbitControlsImpl | null
  const home = useMemo(() => new Vector3(0, 0.7, 0), [])
  const move = useMemo(() => new Vector3(), [])
  useFrame((_, delta) => {
    if (!controls || controls.target.distanceToSquared(home) < 1e-6) return
    move.copy(home).sub(controls.target).multiplyScalar(1 - Math.exp(-Math.min(delta, 0.1) * 3))
    controls.target.add(move)
    controls.object.position.add(move)
  })
  return null
}

function StudioLights() {
  return (
    <>
      <hemisphereLight args={['#ffffff', '#c9ccd2', 0.7]} />
      <directionalLight
        castShadow
        position={[3, 6, 4]}
        intensity={1.8}
        shadow-mapSize={[1024, 1024]}
        shadow-bias={-0.0004}
        shadow-normalBias={0.02}
        shadow-camera-left={-3}
        shadow-camera-right={3}
        shadow-camera-top={3}
        shadow-camera-bottom={-3}
        shadow-camera-near={0.5}
        shadow-camera-far={15}
      />
      {/* Softbox reflections, generated locally so no HDRI download is needed. */}
      <Environment resolution={128} frames={1}>
        <Lightformer form="rect" intensity={2.5} position={[0, 5, -2]} rotation-x={Math.PI / 2} scale={[8, 4, 1]} />
        <Lightformer form="rect" intensity={1.5} position={[-4, 2, 1]} rotation-y={Math.PI / 2} scale={[6, 2, 1]} />
        <Lightformer form="rect" intensity={1.2} position={[4, 1.5, 2]} rotation-y={-Math.PI / 2} scale={[6, 2, 1]} />
        <Lightformer form="ring" intensity={0.8} color="#22d3ee" position={[0, 1, -5]} scale={2} />
      </Environment>
    </>
  )
}

function Floor() {
  return (
    <>
      <RigidBody type="fixed" colliders={false}>
        <CuboidCollider args={[20, 0.5, 20]} position={[0, -0.5, 0]} friction={1.2} />
      </RigidBody>
      <mesh rotation-x={-Math.PI / 2} receiveShadow>
        <planeGeometry args={[60, 60]} />
        <meshStandardMaterial color={BACKGROUND} roughness={0.95} />
      </mesh>
      <Grid
        position={[0, 0.001, 0]}
        infiniteGrid
        cellSize={0.25}
        cellThickness={0.6}
        cellColor="#cfd3da"
        sectionSize={1}
        sectionThickness={1}
        sectionColor="#b4bac4"
        fadeDistance={14}
        fadeStrength={1.5}
      />
    </>
  )
}
