import type { RobotMode } from '../components/Robot'
import { JOINTS, PARTS } from '../robot/skeleton'
import { ACTION_SIZE, CONTROL_DT, OBS_SIZE, REWARD_TERMS } from '../sim/env'

export type StageId = 1 | 2 | 3 | 4 | 5 | 6 | 7

/** Story robot modes, plus `policy`: Byte driven live by the trained network. */
export type StageRobot = RobotMode | 'policy'

export interface Telemetry {
  label: string
  value: string
  tone?: 'ok' | 'warn' | 'bad' | 'info'
}

export interface StageDef {
  id: StageId
  title: string
  text: string
  /** How the physics body is driven during this stage. */
  robotMode: StageRobot
  telemetry: (ctx: { glitching: boolean }) => Telemetry[]
}

const beans = String(PARTS.length)
const joints = String(JOINTS.length)

export const STAGES: StageDef[] = [
  {
    id: 1,
    title: 'Building the Anatomy',
    text: "Human anatomy can be simplified into basic geometric shapes called 'beans'. Let's assemble our robot, Byte, using these shapes and connect them with joints.",
    robotMode: 'assemble',
    telemetry: () => [
      { label: 'beans', value: beans },
      { label: 'joints', value: joints },
      { label: 'colliders', value: 'capsule' },
    ],
  },
  {
    id: 2,
    title: 'Manual Control & Chaos',
    text: 'Without an intelligent policy, Byte is just a physics ragdoll. Let’s apply random motor forces.',
    robotMode: 'chaos',
    telemetry: () => [
      { label: 'policy', value: 'none', tone: 'bad' },
      { label: 'action', value: 'U(lo, hi)', tone: 'warn' },
      { label: 'status', value: 'unstable', tone: 'bad' },
    ],
  },
  {
    id: 3,
    title: 'The High-Poly Upgrade',
    text: 'What if we swap our basic model for a high-poly, 26.9-megabyte rigged character? It usually results in severe physics glitches.',
    robotMode: 'hold',
    telemetry: ({ glitching }) =>
      glitching
        ? [
            { label: 'mesh', value: '26.9 MB', tone: 'bad' },
            { label: 'tris', value: '412,880', tone: 'bad' },
            { label: 'solver', value: 'diverged', tone: 'bad' },
          ]
        : [
            { label: 'mesh', value: 'primitives', tone: 'ok' },
            { label: 'beans', value: beans, tone: 'ok' },
            { label: 'solver', value: 'stable', tone: 'ok' },
          ],
  },
  {
    id: 4,
    title: 'Fixing the Joints for Autonomy',
    text: 'Standard character joints are designed for lifeless ragdolls and cannot accept motor inputs. To prepare Byte for autonomous movement, we must replace every connection with a motorized joint.',
    robotMode: 'motorized',
    telemetry: () => [
      { label: 'joint', value: 'revolute + PD motor', tone: 'info' },
      { label: 'actuated', value: `${joints}/${joints}`, tone: 'ok' },
      { label: 'test', value: 'calibration sweep', tone: 'info' },
    ],
  },
  {
    id: 5,
    title: 'Designing the Rewards',
    text: 'A policy only learns what we reward. Byte earns points for keeping its chest up, holding its waist at walking height, stepping in rhythm with both feet, and closing in on a green goal box. It loses points for falling, jittery motors and slipping feet.',
    robotMode: 'stand',
    telemetry: () => [
      { label: 'terms', value: String(REWARD_TERMS.length), tone: 'info' },
      { label: 'obs', value: String(OBS_SIZE), tone: 'info' },
      { label: 'act', value: String(ACTION_SIZE), tone: 'info' },
    ],
  },
  {
    id: 6,
    title: 'Training the Policy',
    text: 'Byte learned with PPO, practising in hundreds of parallel copies of this exact physics world. Scrub through the checkpoints to watch it go from collapsing on the spot to striding toward the goal.',
    robotMode: 'policy',
    telemetry: () => [
      { label: 'algo', value: 'PPO', tone: 'info' },
      { label: 'net', value: 'MLP 128×128', tone: 'info' },
      { label: 'control', value: `${Math.round(1 / CONTROL_DT)} Hz`, tone: 'info' },
    ],
  },
  {
    id: 7,
    title: 'Goal Navigation',
    text: 'Here is the trained policy steering Byte to green goal boxes across the terrains it has mastered, recorded in the same simulator it trained in. Now you try: click anywhere on the floor to place a goal, and Byte switches to the live neural network to walk there.',
    robotMode: 'policy',
    telemetry: () => [
      { label: 'policy', value: 'trained', tone: 'ok' },
      { label: 'control', value: `${Math.round(1 / CONTROL_DT)} Hz`, tone: 'info' },
      { label: 'physics', value: 'Rapier', tone: 'info' },
    ],
  },
]

export const stageDef = (id: StageId) => STAGES[id - 1]
