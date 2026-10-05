import type { Curriculum, RewardTerms } from '../src/sim/env'

/** PPO hyperparameters (close to the rsl_rl / legged_gym defaults). */
export const PPO = {
  stepsPerEnv: 32,
  epochs: 4,
  minibatches: 4,
  gamma: 0.99,
  lambda: 0.95,
  clip: 0.2,
  valueCoef: 1.0,
  // No entropy bonus: it kept action noise at ~0.5 forever, inflating training
  // falls and stalling the curriculum. PPO's own updates shrink the noise.
  entropyCoef: 0,
  maxGradNorm: 1.0,
  learningRate: 3e-4,
  desiredKl: 0.01,
  initLogStd: -0.7,
  hidden: [128, 128],
} as const

export interface NetParams {
  policy: Float32Array
  value: Float32Array
  logStd: Float32Array
}

export type ToWorker =
  | { type: 'rollout'; nets: NetParams; obsMean: Float32Array; obsStd: Float32Array; curriculum: Curriculum }
  | { type: 'advantageStats' }
  | { type: 'grad'; nets: NetParams; advMean: number; advStd: number; epoch: number; minibatch: number }

export interface EpisodeStats {
  return: number
  seconds: number
  goals: number
  fell: boolean
  terms: RewardTerms
}

export type FromWorker =
  | { type: 'ready' }
  | {
      type: 'rollout'
      samples: number
      obsStats: { count: number; mean: Float64Array; m2: Float64Array }
      episodes: EpisodeStats[]
      /** Goals reached and falls during this rollout (unbiased curriculum signal). */
      goalsReached: number
      falls: number
      rolloutMs: number
    }
  | { type: 'advantageStats'; n: number; sum: number; sumSq: number }
  | {
      type: 'grad'
      n: number
      policy: Float32Array
      value: Float32Array
      logStd: Float32Array
      klSum: number
      clipped: number
      policyLoss: number
      valueLoss: number
    }
