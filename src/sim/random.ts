/** Small, fast, seedable PRNG (mulberry32) so terrain and goals are reproducible. */
export function createRng(seed: number) {
  let a = seed >>> 0
  const next = () => {
    a = (a + 0x6d2b79f5) >>> 0
    let t = a
    t = Math.imul(t ^ (t >>> 15), t | 1)
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61)
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296
  }
  return {
    next,
    range: (lo: number, hi: number) => lo + (hi - lo) * next(),
    int: (lo: number, hiExclusive: number) => lo + Math.floor((hiExclusive - lo) * next()),
    /** Standard normal sample (Box–Muller). */
    normal: () => {
      const u = Math.max(next(), 1e-12)
      return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * next())
    },
  }
}

export type Rng = ReturnType<typeof createRng>
