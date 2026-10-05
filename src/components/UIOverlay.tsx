import { useCallback, useEffect, useRef, useState } from 'react'
import { SITE } from '../config'
import { useLab } from '../story/lab'
import { useStory } from '../story/store'
import { STAGES, stageDef, type StageId, type Telemetry } from '../story/stages'
import { StageExtras } from './LabPanels'

// ---------------------------------------------------------------------------
// Storyline state machine
//
//   1 assemble ─▶ 2 chaos ─▶ 3 glitch ─(auto)─▶ 3 stable ─▶ 4 motorize
//     ─▶ 5 rewards ─▶ 6 training (checkpoint replays) ─▶ 7 live navigation
//        ▲                                                        │
//        └─────────────────────────── replay ─────────────────────┘
//
// Each stage runs a "beat" (its 3D action). "Next" unlocks once the beat is
// complete and the narration has finished typing. Moving forward one stage
// keeps the current physics body so the story flows; any other jump respawns
// Byte from a clean state.
// ---------------------------------------------------------------------------

const GLITCH_MS = 3600

type BeatStep = [atMs: number, run: () => void]
interface BeatActions {
  completeBeat: () => void
  endGlitch: () => void
}

const BEATS: Record<StageId, (a: BeatActions) => BeatStep[]> = {
  1: () => [], // Robot reports completion when the last bean lands.
  2: ({ completeBeat }) => [[3600, completeBeat]],
  3: ({ endGlitch, completeBeat }) => [
    [GLITCH_MS, endGlitch],
    [GLITCH_MS + 900, completeBeat],
  ],
  4: ({ completeBeat }) => [[4000, completeBeat]],
  5: ({ completeBeat }) => [[1800, completeBeat]],
  6: ({ completeBeat }) => [[1500, completeBeat]],
  7: ({ completeBeat }) => [[1500, completeBeat]],
}

const NEXT: Record<StageId, StageId | null> = { 1: 2, 2: 3, 3: 4, 4: 5, 5: 6, 6: 7, 7: null }
const FINAL_STAGE: StageId = 7

function useStoryMachine() {
  const { stage, visit, beatComplete, glitching, enter, completeBeat, endGlitch } = useStory()

  const goTo = useCallback(
    (target: StageId) => {
      const current = useStory.getState().stage
      enter(target, {
        respawn: target !== NEXT[current],
        glitching: target === 3,
      })
    },
    [enter],
  )

  // The RL stages need the published policies; fetch them a stage early.
  useEffect(() => {
    if (stage < 4) return
    const lab = useLab.getState()
    void lab.loadManifest().then(() => {
      const { manifest, selectCheckpoint } = useLab.getState()
      if (!manifest) return
      // Live navigation uses the final policy (stage 6's panel runs its own replay).
      // Each visit starts from the recording and invites the visitor to take over.
      if (stage === 7) {
        useLab.getState().setLiveMode(false)
        void selectCheckpoint(manifest.checkpoints.length - 1)
      }
    })
  }, [stage, visit])

  // Run the current stage's beat; restart it whenever the stage is (re)entered.
  useEffect(() => {
    const timers = BEATS[stage]({ completeBeat, endGlitch }).map(([ms, run]) => window.setTimeout(run, ms))
    return () => timers.forEach(clearTimeout)
  }, [stage, visit, completeBeat, endGlitch])

  const next = NEXT[stage]
  return {
    stage,
    visit,
    beatComplete,
    glitching,
    goTo,
    advance: () => next && goTo(next),
    replay: () => goTo(1),
  }
}

// ---------------------------------------------------------------------------
// Typewriter narration
// ---------------------------------------------------------------------------

function useTypewriter(text: string, resetKey: unknown, charsPerSecond = 52) {
  const [count, setCount] = useState(0)
  const startedAt = useRef(0)

  useEffect(() => {
    if (window.matchMedia('(prefers-reduced-motion: reduce)').matches) {
      setCount(text.length)
      return
    }
    setCount(0)
    startedAt.current = performance.now()
    let raf = 0
    const tick = () => {
      const n = Math.min(text.length, Math.floor(((performance.now() - startedAt.current) / 1000) * charsPerSecond))
      setCount(n)
      if (n < text.length) raf = requestAnimationFrame(tick)
    }
    raf = requestAnimationFrame(tick)
    return () => cancelAnimationFrame(raf)
  }, [text, resetKey, charsPerSecond])

  return {
    shown: text.slice(0, count),
    done: count >= text.length,
    finish: () => {
      startedAt.current = -Infinity
      setCount(text.length)
    },
  }
}

// ---------------------------------------------------------------------------
// Overlay
// ---------------------------------------------------------------------------

export function UIOverlay() {
  const story = useStoryMachine()
  const def = stageDef(story.stage)
  const narration = useTypewriter(def.text, story.visit)
  const isFinal = story.stage === FINAL_STAGE

  const onNext = useCallback(() => {
    if (!narration.done) return narration.finish()
    if (story.beatComplete && !isFinal) story.advance()
  }, [narration, story, isFinal])

  // Keyboard: → or Enter advances (unless a control already has focus).
  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if (e.target instanceof HTMLElement && e.target.closest('button, a, input, textarea')) return
      if (e.key === 'ArrowRight' || e.key === 'Enter') {
        e.preventDefault()
        onNext()
      }
    }
    window.addEventListener('keydown', onKey)
    return () => window.removeEventListener('keydown', onKey)
  }, [onNext])

  const nextReady = !narration.done || story.beatComplete

  return (
    <div className="pointer-events-none absolute inset-0 flex flex-col justify-between p-4 md:p-8">
      {story.glitching && <GlitchEffects />}
      {story.stage === FINAL_STAGE && story.beatComplete && <PlaceGoalHint />}

      {/* Header */}
      <header className="flex items-start justify-between gap-4">
        <div className="pointer-events-auto">
          <h1 className="font-mono text-lg font-bold tracking-tight">{SITE.projectName}</h1>
          <p className="mt-1.5 text-sm text-zinc-600">
            <span className="text-zinc-900">{SITE.creatorName}</span>
          </p>
        </div>
      </header>

      {/* Narration terminal */}
      {/* The story advances by clicking the narration box (controls inside it excepted). */}
      <section
        aria-label="Story"
        onClick={(e) => {
          if (e.target instanceof Element && e.target.closest('button, a, input, label, svg, [role="tab"]')) return
          onNext()
        }}
        className="pointer-events-auto flex cursor-pointer max-h-[calc(100dvh-7.5rem)] w-full max-w-xl flex-col self-start overflow-hidden rounded-2xl border border-white/10 bg-zinc-950/88 text-zinc-100 shadow-2xl shadow-black/20 backdrop-blur-md md:max-h-[calc(100dvh-9rem)]"
      >
        <div className="flex items-center gap-2 border-b border-white/10 px-4 py-2.5 font-mono text-xs text-zinc-400">
          <span className="flex gap-1.5" aria-hidden>
            <span className="size-2.5 rounded-full bg-zinc-700" />
            <span className="size-2.5 rounded-full bg-zinc-700" />
            <span className="size-2.5 rounded-full bg-zinc-700" />
          </span>
          <span className="ml-2 truncate">byte@sim:~/stage-0{story.stage}</span>
          <span className="ml-auto shrink-0 tabular-nums">
            {String(story.stage).padStart(2, '0')} / {String(STAGES.length).padStart(2, '0')}
          </span>
        </div>

        <div className="overflow-y-auto px-5 pt-4 pb-5">
          <h2 className={`font-mono text-xs font-medium tracking-widest text-byte-cyan uppercase ${story.glitching ? 'glitch-text' : ''}`}>
            {def.title}
          </h2>
          <p aria-live="polite" className="mt-2.5 min-h-[4.5rem] text-[15px] leading-relaxed text-zinc-100 md:text-base">
            {narration.shown}
            {!narration.done && <span className="caret ml-0.5 inline-block h-[1.05em] w-2 translate-y-[0.15em] bg-byte-cyan" />}
          </p>

          <TelemetryRow items={def.telemetry({ glitching: story.glitching })} />

          <StageExtras stage={story.stage} />

          <div className="mt-5 flex flex-wrap items-center gap-3">
            {isFinal && story.beatComplete ? (
              <>
                <button
                  type="button"
                  onClick={story.replay}
                  className="ml-auto rounded-lg px-3 py-2.5 font-mono text-sm text-zinc-300 transition hover:bg-white/10 hover:text-white"
                >
                  ↺ Replay story
                </button>
              </>
            ) : (
              <>
                <button
                  type="button"
                  onClick={onNext}
                  disabled={!nextReady}
                  className="inline-flex min-w-32 items-center justify-center gap-2 rounded-lg bg-zinc-100 px-4 py-2.5 font-mono text-sm font-bold text-zinc-950 transition hover:bg-white focus-visible:ring-2 focus-visible:ring-cyan-300 focus-visible:outline-none disabled:cursor-wait disabled:bg-zinc-800 disabled:text-zinc-400"
                >
                  {!narration.done ? 'Skip ▸▸' : story.beatComplete ? 'Next →' : <Simulating />}
                </button>
                <span className="hidden font-mono text-xs text-zinc-500 sm:inline">click to continue · drag the scene to orbit</span>
              </>
            )}
          </div>
        </div>
      </section>
    </div>
  )
}

// ---------------------------------------------------------------------------
// Pieces
// ---------------------------------------------------------------------------

const TONE: Record<NonNullable<Telemetry['tone']>, string> = {
  ok: 'text-emerald-400',
  warn: 'text-amber-400',
  bad: 'text-rose-400',
  info: 'text-cyan-300',
}

function TelemetryRow({ items }: { items: Telemetry[] }) {
  return (
    <dl className="mt-4 flex flex-wrap gap-x-5 gap-y-1.5 font-mono text-xs">
      {items.map((t) => (
        <div key={t.label} className="flex gap-1.5">
          <dt className="text-zinc-500">{t.label}</dt>
          <dd className={t.tone ? TONE[t.tone] : 'text-zinc-200'}>{t.value}</dd>
        </div>
      ))}
    </dl>
  )
}

/** Floating prompt over the 3D scene until the visitor places their first goal. */
function PlaceGoalHint() {
  const placed = useLab((s) => s.goalRequest !== null)
  if (placed) return null
  return (
    <div className="pointer-events-none absolute inset-x-4 top-20 flex justify-center md:inset-x-auto md:top-auto md:right-10 md:bottom-10 md:left-[calc(36rem+3rem)]">
      <div className="animate-bounce rounded-full bg-zinc-950/85 px-4 py-2 font-mono text-xs text-white shadow-lg backdrop-blur">
        ✦ Click the floor to place a goal
      </div>
    </div>
  )
}

function Simulating() {
  return (
    <>
      <span className="size-3 animate-spin rounded-full border-2 border-zinc-500 border-t-byte-cyan" aria-hidden />
      Simulating
    </>
  )
}

const GLITCH_LOG = [
  '> import rigged_character_hp.fbx  (26.9 MB)',
  '  412,880 tris · 1,214 bones · 0 motorized joints',
  '[physics] WARN  convex decomposition → 9,412 colliders',
  '[physics] ERROR contact manifold overflow (body #318)',
  '[physics] ERROR NaN in linvel (bone: spine_03)',
  '[solver]  FATAL constraint graph diverged',
  '> reverting to primitive model…',
]

/** Screen tear plus a scrolling error log while the high-poly model is loaded. */
function GlitchEffects() {
  const [lines, setLines] = useState(1)
  useEffect(() => {
    const step = (GLITCH_MS - 400) / GLITCH_LOG.length
    const id = window.setInterval(() => setLines((n) => Math.min(n + 1, GLITCH_LOG.length)), step)
    return () => clearInterval(id)
  }, [])

  return (
    <>
      <div
        aria-hidden
        className="glitch-tear absolute inset-0 bg-[repeating-linear-gradient(0deg,rgba(255,43,214,0.10)_0px,rgba(255,43,214,0.10)_2px,transparent_2px,transparent_6px)]"
      />
      <pre
        aria-hidden
        className="absolute top-20 right-4 max-w-[calc(100%-2rem)] overflow-hidden rounded-lg bg-zinc-950/85 p-3 font-mono text-[11px] leading-5 whitespace-pre-wrap text-rose-300 md:top-24 md:right-8 md:text-xs"
      >
        {GLITCH_LOG.slice(0, lines).map((line) => (
          <div key={line} className={line.startsWith('>') ? 'text-zinc-300' : ''}>
            {line}
          </div>
        ))}
      </pre>
    </>
  )
}
