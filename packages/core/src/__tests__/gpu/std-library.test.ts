import {describe, expect, it} from 'vitest'
import tgpu from 'typegpu'
import {isWgslLibFn, libSource, linkLibFns, type WgslLibFn} from '@coreroot/std/wgslLibrary'
import {parseWgslDecls} from '@coreroot/std/wgslSource'
import * as library from '@coreroot/std/library/index'

const all: WgslLibFn[] = Object.values(library).filter(isWgslLibFn)

describe('wgsl library: every public function', () => {
    it('exports each function under its own name, once', () => {
        expect(all.length).toBeGreaterThanOrEqual(75)
        for (const [key, fn] of Object.entries(library)) expect(isWgslLibFn(fn) && fn.name).toBe(key)
        expect(new Set(all.map((f) => f.name)).size).toBe(all.length)
    })

    it('documents each function', () => {
        for (const fn of all) {
            expect(fn.signature.startsWith(`${fn.name}(`), fn.name).toBe(true)
            expect(fn.doc.length, fn.name).toBeGreaterThan(20)
            expect(fn.family, fn.name).toMatch(/^[a-z]+$/)
        }
    })

    it.each(all.map((fn) => [fn.name, fn] as const))('%s links and resolves', (name, fn) => {
        const externals = linkLibFns([fn])
        for (const s of fn.structs) expect(externals[s], `${name} exposes struct ${s}`).toBeDefined()
        const wgsl = tgpu.resolve([externals[name] as never], {names: 'strict'})
        expect(wgsl).toMatch(new RegExp(`fn ${name}\\s*\\(`))
        for (const dep of fn.deps) expect(wgsl).toMatch(new RegExp(`fn ${dep.name}\\s*\\(`))
    })

    it('pastes as one module without duplicate declarations (compute modules)', () => {
        const decls = parseWgslDecls(libSource(all))
        const names = [...decls.fns.map((f) => f.name), ...decls.structs.map((s) => s.name), ...decls.consts.map((c) => c.name)]
        expect(new Set(names).size).toBe(names.length)
    })
})

// ── CPU mirrors: JS ports of the same math, for the functions with checkable numbers ──────────

const f32 = (x: number) => Math.fround(x)

function floatBits(x: number): number {
    const view = new DataView(new ArrayBuffer(4))
    view.setFloat32(0, x === 0 ? 0 : x)
    return view.getUint32(0)
}

/** hash11's PCG, in u32 arithmetic. */
function pcg(v: number): number {
    const state = (Math.imul(v, 747796405) + 2891336453) >>> 0
    const word = Math.imul(((state >>> (((state >>> 28) + 4) >>> 0)) ^ state) >>> 0, 277803737) >>> 0
    return ((word >>> 22) ^ word) >>> 0
}
const hash11 = (p: number) => (pcg(floatBits(p)) >>> 8) / 16777216

describe('wgsl library: CPU mirrors', () => {
    it('hash11 is deterministic, in [0, 1), treats -0 as 0, and stays random far from the origin', () => {
        expect(hash11(3)).toBe(hash11(3))
        expect(hash11(-0)).toBe(hash11(0))
        const near = [0, 1, 2, 3, 4, 5, 6, 7].map(hash11)
        const far = [0, 1, 2, 3, 4, 5, 6, 7].map((i) => hash11(1e6 + i))
        for (const v of [...near, ...far]) {
            expect(v).toBeGreaterThanOrEqual(0)
            expect(v).toBeLessThan(1)
        }
        expect(new Set(far).size).toBe(far.length) // adjacent cells far out still differ
        const mean = Array.from({length: 4096}, (_, i) => hash11(i)).reduce((a, b) => a + b, 0) / 4096
        expect(mean).toBeGreaterThan(0.47)
        expect(mean).toBeLessThan(0.53)
    })

    it('easeInOut hits its endpoints and midpoint, and is continuous at 0.5', () => {
        const easeInOut = (t: number) => {
            const x = Math.min(Math.max(t, 0), 1)
            const y = 2 - 2 * x
            return x < 0.5 ? 4 * x * x * x : 1 - y * y * y * 0.5
        }
        expect(easeInOut(0)).toBe(0)
        expect(easeInOut(1)).toBe(1)
        expect(easeInOut(0.5)).toBeCloseTo(0.5, 6)
        expect(easeInOut(0.4999999)).toBeCloseTo(easeInOut(0.5), 5)
        expect(easeInOut(-1)).toBe(0)
        expect(easeInOut(0.25) + easeInOut(0.75)).toBeCloseTo(1, 6)
    })

    it('toOklab / fromOklab round-trip P3-linear colours, and white is L = 1', () => {
        type V3 = [number, number, number]
        const p3ToSrgb = ([r, g, b]: V3): V3 => [
            r * 1.2249401 - g * 0.2249404,
            r * -0.0420569 + g * 1.0420571,
            r * -0.0196376 - g * 0.0786361 + b * 1.0982735,
        ]
        const srgbToP3 = ([r, g, b]: V3): V3 => [
            r * 0.8224621 + g * 0.177538,
            r * 0.0331941 + g * 0.9668058,
            r * 0.0170826 + g * 0.0723974 + b * 0.9105199,
        ]
        const cbrt = (x: number) => Math.sign(x) * Math.pow(Math.max(Math.abs(x), 1e-12), 1 / 3)
        const toOklab = (rgb: V3): V3 => {
            const [r, g, b] = p3ToSrgb(rgb)
            const l = cbrt(r * 0.4122214708 + g * 0.5363325363 + b * 0.0514459929)
            const m = cbrt(r * 0.2119034982 + g * 0.6806995451 + b * 0.1073969566)
            const s = cbrt(r * 0.0883024619 + g * 0.2817188376 + b * 0.6299787005)
            return [
                l * 0.2104542553 + m * 0.793617785 - s * 0.0040720468,
                l * 1.9779984951 - m * 2.428592205 + s * 0.4505937099,
                l * 0.0259040371 + m * 0.7827717662 - s * 0.808675766,
            ]
        }
        const fromOklab = ([L, a, b]: V3): V3 => {
            const l = (L + a * 0.3963377774 + b * 0.2158037573) ** 3
            const m = (L - a * 0.1055613458 - b * 0.0638541728) ** 3
            const s = (L - a * 0.0894841775 - b * 1.291485548) ** 3
            return srgbToP3([
                l * 4.0767416621 - m * 3.3077115913 + s * 0.2309699292,
                l * -1.2684380046 + m * 2.6097574011 - s * 0.3413193965,
                l * -0.0041960863 - m * 0.7034186147 + s * 1.707614701,
            ])
        }
        const white = toOklab([1, 1, 1])
        expect(white[0]).toBeCloseTo(1, 3)
        expect(Math.hypot(white[1], white[2])).toBeLessThan(1e-3)
        for (const c of [[1, 0, 0], [0, 1, 0], [0, 0, 1], [0.2, 0.5, 0.8], [0.01, 0.02, 0.03], [1.5, 0.2, 0]] as V3[]) {
            const back = fromOklab(toOklab(c))
            for (let i = 0; i < 3; i++) expect(back[i]).toBeCloseTo(c[i], 4)
        }
    })

    it('luminance weights are P3 Y and sum to 1', () => {
        expect(f32(0.2289746 + 0.6917385 + 0.0792869)).toBeCloseTo(1, 6)
        expect(library.luminance.code).toContain('0.2289746, 0.6917385, 0.0792869')
    })
})
