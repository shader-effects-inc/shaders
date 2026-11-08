/**
 * `bloom`: the glow of a source's bright parts: pixels above a brightness threshold, extracted
 * and spread with the engine's compute Gaussian. Add it to your colour: `c + halo(uv) * glow`.
 */
import {blur as blurKit} from '../../gpu/kit/index'
import {createComputeModule} from './module'
import {canvasTexture, moduleFor, texelPrelude, uniformBlock} from './common'
import {scalarValue, withBuilder, type InputNode, type InputSource, type ScalarArg} from './types'

export interface BloomInput extends InputNode {
    readonly kind: 'bloom'
    readonly source: InputSource
    readonly threshold: ScalarArg
    readonly radius: ScalarArg
}

/**
 * The glow around a source's bright parts, read by name. `threshold` is the brightness (0–1,
 * linear luminance) where glow starts; `radius` is how far it spreads, in canvas pixels. The
 * result holds only the glow, so add it to your colour.
 *
 * @example
 * ```ts
 * inputs: {halo: bloom(child, {threshold: 0.7, radius: 40})},
 * effect: wgsl`return vec4f(child.rgb + halo(uv).rgb * glow, child.a);`,
 * ```
 * @see blur
 */
export function bloom(source: InputSource, opts: {threshold?: ScalarArg; radius?: ScalarArg} = {}): BloomInput {
    const node = {kind: 'bloom' as const, source, threshold: opts.threshold ?? 0.7, radius: opts.radius ?? 30}
    return withBuilder(node, (ctx) => {
        const input = ctx.source(source)
        if (!input) return null
        // The bright extract at half resolution: glow is soft, so it never needs more.
        const bright = canvasTexture(ctx, {scale: 0.5, max: 1024, label: `${ctx.name} (bright)`})
        const args = uniformBlock(ctx, 1, `${ctx.name}:args`)
        const mod = moduleFor(ctx)
        const extract = createComputeModule(mod, {
            label: `bloom '${ctx.name}'`,
            segments: [],
            entry: `${texelPrelude('_out')}
  let c = _read__src(uv);
  let l = dot(c.rgb, vec3f(0.2126, 0.7152, 0.0722));
  let k = smoothstep(_args.x, _args.x + max(_args.y, 0.001), l);
  textureStore(_out, gid.xy, vec4f(c.rgb * k * c.a, 1.0));`,
            resources: [
                {kind: 'storageTexture', name: '_out', format: 'rgba16float', texture: bright.current},
                {kind: 'uniform', name: '_args', type: 'vec4f', buffer: () => args.buffer},
            ],
            textureReads: [{call: '_src', texture: input}],
            workgroup: [8, 8],
            includeShared: false,
        })
        const extractStep = extract.step(() => [bright.width(), bright.height()])
        let canvas = ctx.canvas()
        const gaussian = blurKit.createGaussianBlurCompute(ctx.root, null, bright.width(), bright.height(), ctx.onCleanup)
        let bound: unknown = null
        ctx.onResize((size) => {
            canvas = size
        })
        return {
            outputs: {
                '': {
                    current: () => gaussian.outputTexture,
                    premultiplied: false,
                    format: 'rgba16float',
                    size: () => ({width: gaussian.computeWidth, height: gaussian.computeHeight}),
                },
            },
            steps() {
                if (!input.current()) return []
                const tex = bright.current()
                if (tex !== bound) {
                    gaussian.setInputTexture(tex as never)
                    gaussian.setInputDimensions(bright.width(), bright.height())
                    bound = tex
                }
                const threshold = scalarValue(node.threshold, ctx.params, 0.7)
                args.write([threshold, 0.15])
                // The radius is in canvas pixels; the extract is at half that resolution.
                const scale = bright.width() / Math.max(1, canvas.width)
                gaussian.updateRadius(Math.max(0, scalarValue(node.radius, ctx.params, 30) * scale))
                return [extractStep, ...gaussian.computeSteps]
            },
        }
    })
}
