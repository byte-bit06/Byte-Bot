# Byte Bot · Sim-to-Real

An interactive 3D case study: a humanoid robot built from primitive "beans",
wired with motorized joints, and trained with reinforcement learning to walk to
a goal. All of it runs in the browser, live, on the same physics it was trained on.

## Run the site

```bash
npm install
npm run dev        # http://localhost:5173  (append ?stage=1..7 to jump to a stage)
npm run build      # production build in dist/
```

## The story

| Stage | What happens | Driven by |
|---|---|---|
| 1 Anatomy | Beans fly in and assemble into Byte | `Robot.tsx` (kinematic) |
| 2 Chaos | Random motor targets; Byte flails and collapses | `Robot.tsx` |
| 3 High-poly glitch | A dense mesh "breaks the solver", then reverts | `GlitchModel.tsx` |
| 4 Motorized joints | Joints come online one by one and run a sweep | `Robot.tsx` |
| 5 Rewards | The reward function, term by term | `src/sim/env.ts` |
| 6 Training | Learning curve + recorded runs of real checkpoints | `ReplayRobot.tsx` |
| 7 Navigation | Final policy on four terrains (recorded), or **Live**: the real network in the browser, click to move the goal | `ReplayRobot.tsx` / `PolicyRobot.tsx` |

Stages 6–7 play back rollouts recorded from the trained policies in the same
simulator (`training/record.ts`), so the site stays smooth on any device. The
Live toggle runs the actual network and physics in the browser instead.

## Layout

```
src/
  robot/skeleton.ts     body plan: parts (beans) and revolute joints — single source of truth
  robot/physics.ts      physics constants shared by the site and the trainer
  sim/byteSim.ts        Byte in a bare Rapier world (no React) — used by training AND the site
  sim/env.ts            RL environment: observations, actions, rewards, termination, curriculum hooks
  sim/terrain.ts        seeded terrain levels (flat → pebbles → rubble → steps)
  sim/policy.ts         MLP + observation normaliser + policy file format
  sim/replay.ts         recorded-clip format (Int16-quantised poses at 20 Hz)
  components/           React Three Fiber scene, story robot, policy robot, UI overlay
training/
  train.ts              PPO trainer (multi-threaded rollouts and gradients)
  worker.ts             rollout + GAE + gradient worker
  nn.ts                 batched backprop and Adam
  publish.ts            copies checkpoints + learning curve into public/policies, records clips
  record.ts             records policy rollouts as compact playback clips
  bench.ts, gradcheck.ts, profile.ts   sanity checks
```

## Train and publish a policy

```bash
npx tsx training/bench.ts                       # physics sanity checks + throughput
npx tsx training/train.ts --run walk-v1         # add --resume to continue a run
npx tsx training/publish.ts --run walk-v1       # → public/policies + recorded clips (~20 s)
```

Training writes to `training/runs/<run>/`: `log.jsonl` (one row per iteration),
`iter_XXXXX.json` checkpoints every 50 iterations, and `state.json` for resuming.

### Environment summary

- **Actions (15):** PD position targets for spine, hip yaw/roll/pitch, knees,
  ankles, shoulders and elbows, as offsets from a slightly crouched stance; 40 Hz control.
- **Observations (61):** gravity, angular and linear velocity in the waist frame,
  goal direction and distance, joint positions and velocities, previous action,
  a gait clock, and foot contacts.
- **Rewards:** chest up, waist height, foot pattern matched to the gait clock
  (both feet down when parked on the goal), air time, potential-based progress to the goal,
  heading, stand-still on the goal, alive bonus, a goal bonus, and penalties for falling,
  crossed feet, action rate, joint speed, joint limits, body wobble and foot slip.
  The weights live in `REWARDS` in `src/sim/env.ts`, and the site renders that same table.
- **Curriculum:** goal distance and angle widen, then domain randomisation
  (motor strength, mass, latency, pushes, sensor noise), then rough terrain.

Personal details shown in the header live in `src/config.ts`.
