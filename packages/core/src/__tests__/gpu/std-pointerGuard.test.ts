import {describe, it, expect} from 'vitest'
import {defineStd, p, simulate, op, pointer, pointerSpeed, displaceBy} from '@coreroot/std'
import {createPointerVelocityTracker} from '@coreroot/gpu/kit/host/pointer'

/**
 * `pointer({teleportGuard})` reaches the wave-field simulation. Both values lower; `'off'`
 * disables the tracker's jump classification so a teleport counts as motion.
 */
function ripples(teleportGuard: 'on' | 'off') {
    const waves = simulate.grid({
        resolution: 64,
        history: 2,
        step: [op.wave({damping: p('decay')}), op.splat({at: pointer({teleportGuard}), amount: pointerSpeed({max: 2}), radius: p('radius')})],
        derive: {displacement: op.gradient()},
    })
    return defineStd({
        name: `Guard${teleportGuard}`,
        role: 'filter',
        species: 'gather',
        props: {decay: {default: 10}, radius: {default: 0.5}, strength: {default: 1}, chromatic: {default: 0}, edges: {default: 'stretch', compileTime: true}},
        effect: displaceBy(waves.output('displacement'), {strength: p('strength'), chromatic: p('chromatic'), edges: p('edges')}),
    })
}

describe("pointer({teleportGuard: 'off'})", () => {
    it('lowers instead of throwing, and keeps the pointer-driven flags', () => {
        expect(() => ripples('off')).not.toThrow()
        expect(ripples('off').usesPointer).toBe(true)
        expect(ripples('on').usesPointer).toBe(true)
    })

    it("'off' is an infinite jump threshold on the tracker: a teleport counts as motion", () => {
        const guarded = createPointerVelocityTracker({initialX: 0.1, initialY: 0.1})
        const jump = guarded.update({x: 0.9, y: 0.9}, 0.016)
        expect(jump.teleport).toBe(true)
        expect(jump.velX).toBe(0)

        const unguarded = createPointerVelocityTracker({initialX: 0.1, initialY: 0.1, teleportGuard: Number.POSITIVE_INFINITY})
        const move = unguarded.update({x: 0.9, y: 0.9}, 0.016)
        expect(move.teleport).toBe(false)
        expect(Math.abs(move.velX)).toBeGreaterThan(0)
    })
})
