/**
 * `blur`: a soft copy of any source (the layer inside, an image or layer prop, another input),
 * computed by the engine's tuned compute Gaussian. Read it in WGSL by its name: `soft(uv)`.
 */
import {blur as blurKit} from '../../gpu/kit/index'
import {
    scalarValue,
    withBuilder,
    type InputNode,
    type InputSource,
    type InputTexture,
    type ScalarArg,
} from './types'

export interface BlurInput extends InputNode {
    readonly kind: 'blur'
    readonly source: InputSource
    readonly radius: ScalarArg
}

/**
 * A blurred copy of a source, read in WGSL by the name you list it under.
 *
 * `radius` is in canvas pixels (a number or a prop). The blur runs at a fixed working
 * resolution, so its cost does not grow with the canvas. Colour comes back with straight alpha,
 * like `child`.
 *
 * @example
 * ```ts
 * inputs: {soft: blur(child, {radius: p('softness')})},
 * effect: wgsl`return mix(child, soft(uv), smoothstep(0.2, 0.6, uv.y));`,
 * ```
 * @see bloom, downsample, child
 */
export function blur(source: InputSource, opts: {radius: ScalarArg}): BlurInput {
    const node = {kind: 'blur' as const, source, radius: opts.radius}
    return withBuilder(node, (ctx) => {
        const input = ctx.source(source)
        if (!input) return null
        let size = input.size()
        const gaussian = blurKit.createGaussianBlurCompute(ctx.root, null, Math.max(1, size.width), Math.max(1, size.height), ctx.onCleanup)
        let bound: unknown = null
        const sync = () => {
            const tex = input.current()
            if (tex && tex !== bound) {
                gaussian.setInputTexture(tex as never)
                bound = tex
            }
            const next = input.size()
            if (next.width !== size.width || next.height !== size.height) {
                size = next
                gaussian.setInputDimensions(Math.max(1, size.width), Math.max(1, size.height))
            }
        }
        const output: InputTexture = {
            current: () => gaussian.outputTexture,
            premultiplied: input.premultiplied,
            format: 'rgba16float',
            size: () => ({width: gaussian.computeWidth, height: gaussian.computeHeight}),
        }
        return {
            outputs: {'': output},
            steps() {
                sync()
                if (!bound) return []
                gaussian.updateRadius(Math.max(0, scalarValue(node.radius, ctx.params, 8)))
                return gaussian.computeSteps
            },
            bindLate() {
                sync()
            },
        }
    })
}
