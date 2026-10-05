import { create } from 'zustand'
import type { StageId } from './stages'

/**
 * Story state shared by the DOM overlay (which drives it) and the 3D scene
 * (which renders it). The transition rules live in UIOverlay.
 */
interface StoryState {
  stage: StageId
  /** Increments on every stage entry, so re-entering a stage restarts its beat. */
  visit: number
  /** The current stage's 3D action has finished; "Next" may proceed. */
  beatComplete: boolean
  /** The high-poly stand-in is on screen instead of Byte. */
  glitching: boolean
  /** Changing this remounts Byte from a clean physics state. */
  robotKey: number

  enter: (stage: StageId, opts: { respawn: boolean; glitching: boolean }) => void
  completeBeat: () => void
  endGlitch: () => void
}

/** `?stage=3` deep-links straight into a stage. */
function initialStage(): StageId {
  const n = Number(new URLSearchParams(window.location.search).get('stage'))
  return n >= 1 && n <= 7 ? (n as StageId) : 1
}

const startStage = initialStage()

export const useStory = create<StoryState>()((set) => ({
  stage: startStage,
  visit: 0,
  beatComplete: false,
  glitching: startStage === 3,
  robotKey: 0,

  enter: (stage, { respawn, glitching }) =>
    set((s) => ({
      stage,
      visit: s.visit + 1,
      beatComplete: false,
      glitching,
      robotKey: respawn ? s.robotKey + 1 : s.robotKey,
    })),
  completeBeat: () => set({ beatComplete: true }),
  endGlitch: () => set((s) => ({ glitching: false, robotKey: s.robotKey + 1 })),
}))
