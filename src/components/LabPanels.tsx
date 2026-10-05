import { memo, useEffect, useMemo, useRef, useState } from 'react'
import { REWARDS, REWARD_TERMS, type RewardTerm } from '../sim/env'
import { TERRAIN_LEVELS } from '../sim/terrain'
import { useLab, type CurvePoint } from '../story/lab'
import type { StageId } from '../story/stages'

/** Stage-specific widgets under the narration for the RL stages (memoised: the typewriter re-renders its parent every frame). */
export const StageExtras = memo(function StageExtras({ stage }: { stage: StageId }) {
  if (stage === 5) return <RewardTable />
  if (stage === 6) return <TrainingPanel />
  if (stage === 7) return <NavigationPanel />
  return null
})

// ---------------------------------------------------------------------------
// Stage 5: the reward function
// ---------------------------------------------------------------------------

function RewardTable() {
  const rows = REWARD_TERMS.map((k) => ({ key: k, ...REWARDS[k] }))
  const positive = rows.filter((r) => r.weight > 0)
  const negative = rows.filter((r) => r.weight < 0)
  return (
    <div className="mt-4 grid gap-3 sm:grid-cols-2">
      <RewardList title="Rewards" rows={positive} />
      <RewardList title="Penalties" rows={negative} />
    </div>
  )
}

function RewardList({
  title,
  rows,
}: {
  title: string
  rows: { key: string; label: string; note: string; weight: number; event?: boolean }[]
}) {
  return (
    <section className="rounded-lg border border-white/10 bg-white/[0.03] p-3">
      <h3 className="font-mono text-[11px] tracking-widest text-zinc-400 uppercase">{title}</h3>
      <ul className="mt-2 space-y-1.5">
        {rows.map((r) => (
          <li key={r.key} className="text-xs leading-snug" title={r.note}>
            <div className="flex items-baseline justify-between gap-2">
              <span className="text-zinc-100">{r.label}</span>
              <span className="shrink-0 font-mono text-zinc-400 tabular-nums">
                {r.weight > 0 ? '+' : ''}
                {r.weight}
                {r.event ? '' : '/s'}
              </span>
            </div>
            <div className="text-[11px] text-zinc-500">{r.note}</div>
          </li>
        ))}
      </ul>
    </section>
  )
}

// ---------------------------------------------------------------------------
// Stage 6: learning curve + checkpoint scrubber
// ---------------------------------------------------------------------------

const METRICS = {
  meanReturn: { label: 'Episode return', format: (v: number) => v.toFixed(1) },
  goalsPerEpisode: { label: 'Goals per episode', format: (v: number) => v.toFixed(2) },
  fallRate: { label: 'Fall rate', format: (v: number) => `${Math.round(v * 100)}%` },
} as const
type Metric = keyof typeof METRICS

const AUTOPLAY_MS = 6000

function TrainingPanel() {
  const { status, manifest, checkpoint, selectCheckpoint } = useLab()
  const [metric, setMetric] = useState<Metric>('meanReturn')
  const [playing, setPlaying] = useState(true)

  // Replays start from the untrained network.
  useEffect(() => {
    if (manifest) void selectCheckpoint(0)
  }, [manifest, selectCheckpoint])

  // Step through checkpoints automatically until the visitor takes over.
  useEffect(() => {
    if (!playing || !manifest) return
    const last = manifest.checkpoints.length - 1
    if (checkpoint >= last) return
    const id = window.setTimeout(() => {
      void selectCheckpoint(checkpoint + 1)
      if (checkpoint + 1 === last) setPlaying(false)
    }, AUTOPLAY_MS)
    return () => clearTimeout(id)
  }, [playing, manifest, checkpoint, selectCheckpoint])

  if (status !== 'ready' || !manifest) return <PolicyStatus status={status} />
  const current = manifest.checkpoints[checkpoint]

  return (
    <div className="mt-4">
      <div className="flex flex-wrap items-center gap-1.5" role="tablist" aria-label="Metric">
        {(Object.keys(METRICS) as Metric[]).map((m) => (
          <button
            key={m}
            type="button"
            role="tab"
            aria-selected={metric === m}
            onClick={() => setMetric(m)}
            className={`rounded-md px-2 py-1 font-mono text-[11px] transition ${
              metric === m ? 'bg-white/15 text-white' : 'text-zinc-400 hover:bg-white/5 hover:text-zinc-200'
            }`}
          >
            {METRICS[m].label}
          </button>
        ))}
      </div>

      <LearningCurve
        curve={manifest.curve}
        metric={metric}
        markers={manifest.checkpoints.map((c) => c.envSteps)}
        selected={current.envSteps}
      />

      <div className="mt-3 flex items-center gap-3">
        <button
          type="button"
          onClick={() => {
            if (!playing && checkpoint === manifest.checkpoints.length - 1) void selectCheckpoint(0)
            setPlaying((p) => !p)
          }}
          className="grid size-8 shrink-0 place-items-center rounded-md bg-white/10 font-mono text-xs text-white hover:bg-white/20"
          aria-label={playing ? 'Pause checkpoint playback' : 'Play checkpoints'}
        >
          {playing ? '❚❚' : '▶'}
        </button>
        <input
          type="range"
          min={0}
          max={manifest.checkpoints.length - 1}
          value={checkpoint}
          onChange={(e) => {
            setPlaying(false)
            void selectCheckpoint(Number(e.target.value))
          }}
          className="w-full accent-cyan-400"
          aria-label="Checkpoint"
        />
      </div>
      <dl className="mt-2 flex flex-wrap gap-x-4 gap-y-1 font-mono text-[11px]">
        <Stat label="iteration" value={String(current.iteration)} />
        <Stat label="experience" value={formatSteps(current.envSteps)} />
        <Stat label="sim time" value={formatSimTime(current.envSteps)} />
        <Stat label="goals/ep" value={current.goalsPerEpisode.toFixed(2)} />
      </dl>
    </div>
  )
}

function Stat({ label, value }: { label: string; value: string }) {
  return (
    <div className="flex gap-1.5">
      <dt className="text-zinc-500">{label}</dt>
      <dd className="text-zinc-200 tabular-nums">{value}</dd>
    </div>
  )
}

const formatSteps = (n: number) => (n >= 1e6 ? `${(n / 1e6).toFixed(1)}M steps` : `${Math.round(n / 1e3)}k steps`)
/** Each policy step is 1/40 s of simulated experience. */
const formatSimTime = (steps: number) => {
  const hours = steps / 40 / 3600
  return hours >= 24 ? `${(hours / 24).toFixed(1)} days` : `${hours.toFixed(1)} h`
}

function PolicyStatus({ status }: { status: string }) {
  return (
    <p className="mt-4 rounded-lg border border-amber-400/30 bg-amber-400/10 p-3 font-mono text-xs text-amber-200">
      {status === 'missing'
        ? 'No trained policy published yet. Run the trainer, then `npx tsx training/publish.ts`.'
        : 'Loading policy…'}
    </p>
  )
}

/** Single-series line chart with checkpoint dots and a hover crosshair. */
function LearningCurve({
  curve,
  metric,
  markers,
  selected,
}: {
  curve: CurvePoint[]
  metric: Metric
  markers: number[]
  selected: number
}) {
  const W = 480
  const H = 140
  const pad = { l: 36, r: 8, t: 8, b: 20 }
  const svgRef = useRef<SVGSVGElement>(null)
  const [hover, setHover] = useState<number | null>(null)

  const geo = useMemo(() => {
    const xs = curve.map((p) => p.envSteps)
    const ys = curve.map((p) => p[metric])
    const xMax = Math.max(1, ...xs)
    let yMin = Math.min(0, ...ys)
    let yMax = Math.max(...ys)
    if (metric === 'fallRate') {
      yMin = 0
      yMax = 1
    }
    if (yMax === yMin) yMax = yMin + 1
    const x = (v: number) => pad.l + (v / xMax) * (W - pad.l - pad.r)
    const y = (v: number) => pad.t + (1 - (v - yMin) / (yMax - yMin)) * (H - pad.t - pad.b)
    const path = curve.map((p, i) => `${i ? 'L' : 'M'}${x(p.envSteps).toFixed(1)},${y(p[metric]).toFixed(1)}`).join('')
    const ticks = [yMin, (yMin + yMax) / 2, yMax]
    const valueAt = (steps: number) => {
      let best = curve[0]
      for (const p of curve) if (Math.abs(p.envSteps - steps) < Math.abs(best.envSteps - steps)) best = p
      return best
    }
    return { x, y, path, ticks, xMax, valueAt }
  }, [curve, metric, pad.b, pad.l, pad.r, pad.t])

  const fmt = METRICS[metric].format
  const hovered = hover != null ? geo.valueAt(hover) : null

  const onMove = (e: React.PointerEvent) => {
    const rect = svgRef.current!.getBoundingClientRect()
    const px = ((e.clientX - rect.left) / rect.width) * W
    const steps = ((px - pad.l) / (W - pad.l - pad.r)) * geo.xMax
    setHover(Math.max(0, Math.min(geo.xMax, steps)))
  }

  return (
    <figure className="relative mt-2">
      <figcaption className="sr-only">
        {METRICS[metric].label} over training, from {fmt(curve[0]?.[metric] ?? 0)} to{' '}
        {fmt(curve[curve.length - 1]?.[metric] ?? 0)}.
      </figcaption>
      <svg
        ref={svgRef}
        viewBox={`0 0 ${W} ${H}`}
        className="w-full touch-none select-none"
        onPointerMove={onMove}
        onPointerLeave={() => setHover(null)}
        role="img"
        aria-label={`${METRICS[metric].label} learning curve`}
      >
        {geo.ticks.map((t) => (
          <g key={t}>
            <line x1={pad.l} x2={W - pad.r} y1={geo.y(t)} y2={geo.y(t)} stroke="rgba(255,255,255,0.08)" />
            <text x={pad.l - 6} y={geo.y(t) + 3} textAnchor="end" className="fill-zinc-500 font-mono text-[9px]">
              {fmt(t)}
            </text>
          </g>
        ))}
        <text x={W - pad.r} y={H - 4} textAnchor="end" className="fill-zinc-500 font-mono text-[9px]">
          {formatSteps(geo.xMax)}
        </text>
        <path d={geo.path} fill="none" stroke="#22d3ee" strokeWidth={2} strokeLinejoin="round" />
        {markers.map((m) => {
          const p = geo.valueAt(m)
          const isSel = m === selected
          return (
            <circle
              key={m}
              cx={geo.x(m)}
              cy={geo.y(p[metric])}
              r={isSel ? 5.5 : 4}
              fill={isSel ? '#22d3ee' : '#09090b'}
              stroke={isSel ? '#09090b' : '#22d3ee'}
              strokeWidth={2}
            />
          )
        })}
        {hovered && (
          <g pointerEvents="none">
            <line
              x1={geo.x(hovered.envSteps)}
              x2={geo.x(hovered.envSteps)}
              y1={pad.t}
              y2={H - pad.b}
              stroke="rgba(255,255,255,0.3)"
            />
            <circle cx={geo.x(hovered.envSteps)} cy={geo.y(hovered[metric])} r={4} fill="#22d3ee" stroke="#09090b" strokeWidth={2} />
          </g>
        )}
      </svg>
      {hovered && (
        <div
          className="pointer-events-none absolute top-0 rounded-md bg-zinc-800 px-2 py-1 font-mono text-[11px] whitespace-nowrap text-zinc-100 shadow-lg"
          style={{
            left: `${(geo.x(hovered.envSteps) / W) * 100}%`,
            transform: `translateX(${geo.x(hovered.envSteps) > W * 0.6 ? '-105%' : '5%'})`,
          }}
        >
          <div>{fmt(hovered[metric])}</div>
          <div className="text-zinc-400">
            iter {hovered.iteration} · {formatSteps(hovered.envSteps)}
          </div>
        </div>
      )}
    </figure>
  )
}

// ---------------------------------------------------------------------------
// Stage 7: live navigation
// ---------------------------------------------------------------------------

function NavigationPanel() {
  const { status, manifest, terrainLevel, setTerrainLevel, live, liveMode, setLiveMode, goalRequest } = useLab()
  if (status !== 'ready') return <PolicyStatus status={status} />

  // Biggest live contributors, either sign.
  const terms = REWARD_TERMS.filter((k) => !(REWARDS[k] as { event?: boolean }).event)
    .map((k) => ({ key: k, rate: live.rates[k] }))
    .sort((a, b) => Math.abs(b.rate) - Math.abs(a.rate))
    .slice(0, 7)
  const scale = Math.max(0.5, ...terms.map((t) => Math.abs(t.rate)))

  return (
    <div className="mt-4">
      <YourTurn placed={goalRequest !== null} live={liveMode} goals={live.goals} />
      <div className="mb-2 flex flex-wrap items-center gap-1.5">
        <span className="mr-1 font-mono text-[11px] text-zinc-500">mode</span>
        {[
          { on: false, label: 'Recorded', hint: 'smooth playback' },
          { on: true, label: 'Live', hint: 'real network, heavier' },
        ].map((m) => (
          <button
            key={m.label}
            type="button"
            onClick={() => setLiveMode(m.on)}
            aria-pressed={liveMode === m.on}
            title={m.hint}
            className={`rounded-md px-2 py-1 font-mono text-[11px] transition ${
              liveMode === m.on ? 'bg-white/15 text-white' : 'text-zinc-400 hover:bg-white/5 hover:text-zinc-200'
            }`}
          >
            {m.label}
          </button>
        ))}
        <span className="font-mono text-[11px] text-zinc-500">
          {liveMode ? '· click the floor to move the goal' : '· playback of the trained policy'}
        </span>
      </div>
      <div className="flex flex-wrap items-center gap-1.5">
        <span className="mr-1 font-mono text-[11px] text-zinc-500">terrain</span>
        {(manifest?.navigation ?? []).map(({ terrainLevel: i }) => (
          <button
            key={i}
            type="button"
            onClick={() => setTerrainLevel(i)}
            aria-pressed={terrainLevel === i}
            className={`rounded-md px-2 py-1 font-mono text-[11px] capitalize transition ${
              terrainLevel === i ? 'bg-white/15 text-white' : 'text-zinc-400 hover:bg-white/5 hover:text-zinc-200'
            }`}
          >
            {TERRAIN_LEVELS[i].name}
          </button>
        ))}
      </div>

      <dl className="mt-3 flex flex-wrap gap-x-4 gap-y-1 font-mono text-[11px]">
        <Stat label="goals reached" value={String(live.goals)} />
        <Stat label="falls" value={String(live.falls)} />
        <Stat label="upright for" value={`${live.upFor.toFixed(0)} s`} />
      </dl>

      <h3 className="mt-3 font-mono text-[11px] tracking-widest text-zinc-400 uppercase">Live reward / second</h3>
      <ul className="mt-1.5 space-y-1">
        {terms.map(({ key, rate }) => (
          <RewardBar key={key} term={key} rate={rate} scale={scale} />
        ))}
      </ul>
    </div>
  )
}

/** Invites the visitor to place a goal themselves, then confirms they're driving the live network. */
function YourTurn({ placed, live, goals }: { placed: boolean; live: boolean; goals: number }) {
  if (placed && live) {
    return (
      <p className="mb-3 rounded-lg border border-emerald-400/30 bg-emerald-400/10 px-3 py-2 text-xs leading-relaxed text-emerald-100">
        <span className="font-semibold">You're in control.</span> The neural network is running live in your browser
        {goals > 0 ? ` and has reached ${goals} of your goal${goals === 1 ? '' : 's'}` : ''}. Click anywhere else to send
        Byte somewhere new.
      </p>
    )
  }
  return (
    <p className="mb-3 flex items-start gap-2 rounded-lg border border-cyan-400/40 bg-cyan-400/10 px-3 py-2 text-xs leading-relaxed text-cyan-50">
      <span className="mt-0.5 inline-block size-2 shrink-0 animate-ping rounded-full bg-cyan-300" aria-hidden />
      <span>
        <span className="font-semibold">Your turn:</span> click anywhere on the floor to place a goal. Byte will switch
        to the live neural network and walk to it.
      </span>
    </p>
  )
}

/** Diverging bar from a centre zero line: right = reward, left = penalty. */
function RewardBar({ term, rate, scale }: { term: RewardTerm; rate: number; scale: number }) {
  const width = Math.min(50, (Math.abs(rate) / scale) * 50)
  return (
    <li className="grid grid-cols-[7.5rem_1fr_3rem] items-center gap-2 text-[11px]">
      <span className="truncate text-zinc-300">{REWARDS[term].label}</span>
      <span className="relative h-2 rounded-sm bg-white/5">
        <span className="absolute top-0 left-1/2 h-full w-px bg-white/20" />
        <span
          className={`absolute top-0 h-full rounded-sm ${rate >= 0 ? 'bg-cyan-400' : 'bg-rose-400'}`}
          style={rate >= 0 ? { left: '50%', width: `${width}%` } : { right: '50%', width: `${width}%` }}
        />
      </span>
      <span className="text-right font-mono text-zinc-300 tabular-nums">{rate.toFixed(2)}</span>
    </li>
  )
}
