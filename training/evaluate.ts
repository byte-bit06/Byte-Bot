/**
 * Deterministic evaluation of a policy, exactly as the website runs it.
 *
 *   npx tsx training/evaluate.ts --run walk-v1 [--file latest.json] [--seeds 8] [--seconds 30]
 */
import { readFileSync } from 'node:fs'
import { join } from 'node:path'
import { parseArgs } from 'node:util'
import RAPIER from '@dimforge/rapier3d-compat'
import type { PolicyFile } from '../src/sim/policy'
import { TERRAIN_LEVELS } from '../src/sim/terrain'
import { recordClip } from './record'

const { values: args } = parseArgs({
  options: {
    run: { type: 'string', default: 'walk-v1' },
    file: { type: 'string', default: 'latest.json' },
    seeds: { type: 'string', default: '8' },
    seconds: { type: 'string', default: '30' },
  },
})

await RAPIER.init()
const policy = JSON.parse(readFileSync(join('training', 'runs', args.run!, args.file!), 'utf8')) as PolicyFile
console.log(`${args.file}: iteration ${policy.meta.iteration}, ${(policy.meta.envSteps / 1e6).toFixed(1)}M steps`)

const seconds = Number(args.seconds)
for (let level = 0; level < TERRAIN_LEVELS.length; level++) {
  let goals = 0
  let falls = 0
  let clean = 0
  const n = Number(args.seeds)
  for (let s = 0; s < n; s++) {
    const r = recordClip(RAPIER, policy, { label: '', seconds, terrainLevel: level, terrainSeed: 2024 + s, seed: 100 + s })
    goals += r.goals
    falls += r.falls
    if (r.falls === 0) clean++
  }
  console.log(
    `  ${TERRAIN_LEVELS[level].name.padEnd(8)} goals/min ${((goals / n / seconds) * 60).toFixed(1).padStart(5)} | ` +
      `falls/min ${((falls / n / seconds) * 60).toFixed(2)} | fall-free runs ${clean}/${n}`,
  )
}
