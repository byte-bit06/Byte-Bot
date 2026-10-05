/**
 * Copy a training run's checkpoints and learning curve into public/policies
 * for the website, and record playback clips of each policy (the site plays
 * these instead of simulating live, which keeps it smooth on any device).
 *
 *   npx tsx training/publish.ts --run walk-v1 [--checkpoints 8]
 */
import { readFileSync, readdirSync, writeFileSync, mkdirSync, rmSync, existsSync } from 'node:fs'
import { join } from 'node:path'
import RAPIER from '@dimforge/rapier3d-compat'
import { parseArgs } from 'node:util'
import type { PolicyFile } from '../src/sim/policy'
import { encodeFloat32, decodeFloat32 } from '../src/sim/policy'
import type { CheckpointInfo, CurvePoint, Manifest, NavigationClip } from '../src/story/lab'
import { TERRAIN_LEVELS } from '../src/sim/terrain'
import { recordBest, recordClip } from './record'

await RAPIER.init()

const { values: args } = parseArgs({
  options: {
    run: { type: 'string', default: 'walk-v1' },
    checkpoints: { type: 'string', default: '8' },
  },
})

const RUN_DIR = join('training', 'runs', args.run!)
const OUT_DIR = join('public', 'policies')

const files = readdirSync(RUN_DIR)
  .filter((f) => /^iter_\d+\.json$/.test(f))
  .sort()
if (!files.length) throw new Error(`no checkpoints in ${RUN_DIR} yet`)

// Evenly spaced checkpoints, always including the newest.
const wanted = Math.min(Number(args.checkpoints), files.length)
const picked = new Set<string>()
for (let i = 0; i < wanted; i++) picked.add(files[Math.round((i * (files.length - 1)) / Math.max(1, wanted - 1))])
picked.add(files[files.length - 1])

const load = (f: string) => JSON.parse(readFileSync(join(RUN_DIR, f), 'utf8')) as PolicyFile

rmSync(OUT_DIR, { recursive: true, force: true })
mkdirSync(OUT_DIR, { recursive: true })

mkdirSync(join(OUT_DIR, 'clips'), { recursive: true })
const writeJson = (name: string, data: unknown) => writeFileSync(join(OUT_DIR, name), JSON.stringify(data))

const checkpoints: CheckpointInfo[] = []
const write = (name: string, policy: PolicyFile) => {
  writeJson(name, policy)
  // One honest clip per checkpoint (fixed seed, no cherry-picking) for the training replay.
  const clipFile = `clips/${name}`
  const { clip } = recordClip(RAPIER, policy, {
    label: `iteration ${policy.meta.iteration}`,
    seconds: 10,
    terrainLevel: 0,
    terrainSeed: 2024,
    seed: 7,
  })
  writeJson(clipFile, clip)
  checkpoints.push({
    clip: clipFile,
    file: name,
    iteration: policy.meta.iteration,
    envSteps: policy.meta.envSteps,
    wallSeconds: policy.meta.wallSeconds,
    meanReturn: policy.meta.meanReturn,
    goalsPerEpisode: policy.meta.goalsPerEpisode,
    fallRate: policy.meta.fallRate,
  })
}

// "Iteration 0": the untrained network. Its output layer starts near zero, so
// it commands the default stance and nothing else — exactly what this is.
const first = load([...picked][0])
const untrained = new Float32Array(decodeFloat32(first.params).length)
write('ckpt-00000.json', {
  ...first,
  params: encodeFloat32(untrained),
  meta: { ...first.meta, iteration: 0, envSteps: 0, wallSeconds: 0, meanReturn: 0, goalsPerEpisode: 0, fallRate: 1 },
})
for (const f of [...picked].sort()) {
  const policy = load(f)
  write(`ckpt-${String(policy.meta.iteration).padStart(5, '0')}.json`, policy)
}

// Learning curve, downsampled to ≤ 300 points.
const logPath = join(RUN_DIR, 'log.jsonl')
const rows = existsSync(logPath)
  ? readFileSync(logPath, 'utf8')
      .trim()
      .split('\n')
      .map((l) => JSON.parse(l))
  : []
const stride = Math.max(1, Math.ceil(rows.length / 300))
const curve: CurvePoint[] = rows
  .filter((_, i) => i % stride === 0 || i === rows.length - 1)
  .map((r) => ({
    iteration: r.iteration,
    envSteps: r.envSteps,
    meanReturn: +r.meanReturn.toFixed(3),
    goalsPerEpisode: +r.goalsPerEpisode.toFixed(3),
    fallRate: +r.fallRate.toFixed(3),
    episodeSeconds: +r.meanSeconds.toFixed(2),
    level: r.level,
  }))

// Navigation showcase: the final policy on each terrain it has mastered. A
// terrain is only offered once a 30 s run has no falls; the rest unlock on a
// later publish as training improves.
const final = load([...picked].sort().at(-1)!)
const navigation: NavigationClip[] = []
TERRAIN_LEVELS.forEach((level, i) => {
  const { clip, goals, falls } = recordBest(
    RAPIER,
    final,
    { label: level.name, seconds: 30, terrainLevel: i, terrainSeed: 2024 },
    6,
  )
  const ok = falls === 0 || i === 0
  console.log(`  navigation on ${level.name}: ${goals} goals, ${falls} falls in 30 s${ok ? '' : ' — not published yet'}`)
  if (!ok) return
  const file = `clips/nav-${i}.json`
  writeJson(file, clip)
  navigation.push({ terrainLevel: i, file, goals, falls })
})

const manifest: Manifest = { run: args.run!, checkpoints, curve, navigation }
writeFileSync(join(OUT_DIR, 'manifest.json'), JSON.stringify(manifest))
console.log(
  `published ${checkpoints.length} checkpoints (${checkpoints.map((c) => c.iteration).join(', ')}) and ${curve.length} curve points → ${OUT_DIR}`,
)
