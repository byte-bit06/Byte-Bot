import { Suspense, useState } from 'react'
import { Canvas } from '@react-three/fiber'
import { PerformanceMonitor } from '@react-three/drei'
import { Scene } from './components/Scene'
import { UIOverlay } from './components/UIOverlay'

const MAX_DPR = Math.min(typeof window === 'undefined' ? 1 : window.devicePixelRatio, 1.5)

export default function App() {
  // Start crisp, drop resolution if the frame rate can't keep up.
  const [dpr, setDpr] = useState(MAX_DPR)
  return (
    <main className="relative h-dvh w-full overflow-hidden bg-byte-bg font-sans text-byte-ink">
      <Canvas
        shadows="percentage"
        dpr={dpr}
        gl={{ antialias: true, powerPreference: 'high-performance' }}
        camera={{ position: [2.3, 1.55, 3.1], fov: 40, near: 0.05, far: 60 }}
        className="absolute! inset-0"
      >
        <PerformanceMonitor onDecline={() => setDpr(1)} onIncline={() => setDpr(MAX_DPR)} flipflops={3} onFallback={() => setDpr(1)} />
        <Suspense fallback={null}>
          <Scene />
        </Suspense>
      </Canvas>
      <UIOverlay />
    </main>
  )
}
