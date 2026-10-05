export const clamp = (v: number, lo: number, hi: number) => Math.min(hi, Math.max(lo, v))
export const clamp01 = (v: number) => clamp(v, 0, 1)
export const lerp = (a: number, b: number, t: number) => a + (b - a) * t
export const randRange = (lo: number, hi: number) => lerp(lo, hi, Math.random())

export const smoothstep = (t: number) => {
  const x = clamp01(t)
  return x * x * (3 - 2 * x)
}

export const easeInOutCubic = (t: number) => {
  const x = clamp01(t)
  return x < 0.5 ? 4 * x * x * x : 1 - (-2 * x + 2) ** 3 / 2
}

/** Ease-out with a small overshoot, so parts "snap" into their sockets. */
export const easeOutBack = (t: number, overshoot = 1.4) => {
  const x = clamp01(t)
  const c3 = overshoot + 1
  return 1 + c3 * (x - 1) ** 3 + overshoot * (x - 1) ** 2
}
