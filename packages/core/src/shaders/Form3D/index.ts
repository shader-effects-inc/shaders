import type {GpuShaderDefinition, GpuFragmentParams, Expr} from "@coreroot/gpu/porters"
import {call, ZERO, animatedTime} from "@coreroot/gpu/porters"
import {defineShader} from "@coreroot/std"
import {blend, effects, sdf3d, constants} from "@coreroot/gpu/kit"
import {transformPosition, transformEdges} from "@coreroot/utilities/transformations"

const {applyUVMode3d, distortion3dLighting, buildParamThreadedTrace} = effects.raymarch3d

const DEG = constants.DEG_TO_RAD
const TAU_RATE = (Math.PI * 2) / 10 // spin-rate scaling (10× reduction)
const DEFAULT_RIBBON = JSON.stringify({ type: 'ribbon', angle: 0, twist: 50, width: 40, thickness: 20, seed: 0 })

// ─────────────────────────────────────────────────────────────────────────────
// Shape atlas: the kit's param-threaded raymarch primitives (sdf3d section E). The rotation /
// spin / size params ride in as packed vec4 fn-args (rp0/rp1/rp2 + t) threaded through the trace
// to the sdf each march step; the CPU driver below writes them per frame as extraFields. Packing
// (rotation shapes):
//   rp0 = (cx, sx, cy, sy)   rp1 = (cz, sz, sizeA, sizeB)   rp2 = (sizeC, sizeD, sizeE, _)
// Ribbon (no 3D rotation): rp0 = (angleRad, twist, halfW, width), rp1 = (thickness, phase, lipschitz, _), t = animTime.
// ─────────────────────────────────────────────────────────────────────────────

// A raw shape SDF tgpu.fn: `(p, rp0, rp1, rp2, t) => f32` — called DIRECTLY inside the kit trace
// body (a `'use gpu'` body invokes tgpu.fns directly, never the builder-level `call`).
type ShapeSdf = effects.raymarch3d.ParamThreadedSdf
type ShapeFns = {
    sdf: ShapeSdf
    surfaceUV: unknown // tgpu.fn, invoked at builder level via `call(...)`
    uvHint: string     // distinct WGSL name for the surfaceUV call (avoids cross-shape collision)
    steps: number; stepCap: number; hitEps: number
}
const SHAPES: Record<string, ShapeFns> = {
    sphere:  {sdf: sdf3d.paramSphereSdf as unknown as ShapeSdf,        surfaceUV: sdf3d.sphereSurfaceUV,        uvHint: 'sphereSurfaceUV',        steps: 64,  stepCap: 1.5,  hitEps: 0.001},
    torus:   {sdf: sdf3d.paramTorusSdf as unknown as ShapeSdf,         surfaceUV: sdf3d.torusSurfaceUV,         uvHint: 'torusSurfaceUV',         steps: 96,  stepCap: 0.5,  hitEps: 0.001},
    box:     {sdf: sdf3d.paramBoxSdf as unknown as ShapeSdf,           surfaceUV: sdf3d.boxFaceUV,              uvHint: 'boxFaceUV',              steps: 96,  stepCap: 0.5,  hitEps: 0.001},
    capsule: {sdf: sdf3d.paramCapsuleSdf as unknown as ShapeSdf,       surfaceUV: sdf3d.capsuleSurfaceUV,       uvHint: 'capsuleSurfaceUV',       steps: 64,  stepCap: 1.5,  hitEps: 0.001},
    mobius:  {sdf: sdf3d.paramMobiusSdf as unknown as ShapeSdf,        surfaceUV: sdf3d.mobiusSurfaceUV,        uvHint: 'mobiusSurfaceUV',        steps: 96,  stepCap: 0.3,  hitEps: 0.001},
    ribbon:  {sdf: sdf3d.paramTwistedRibbonSdf as unknown as ShapeSdf, surfaceUV: sdf3d.twistedRibbonSurfaceUV, uvHint: 'twistedRibbonSurfaceUV', steps: 128, stepCap: 0.08, hitEps: 0.002},
}

// ─────────────────────────────────────────────────────────────────────────────
// CPU shape-param driver: parse the shape JSON each frame, accumulate spin, precompute rotation
// cos/sin, scale sizes, and write the packed rp0/rp1/rp2 extraFields.
// ─────────────────────────────────────────────────────────────────────────────
interface RotSpinDefaults { rotX?: number; rotY?: number; rotZ?: number; spinX?: number; spinY?: number; spinZ?: number }
const SPIN_DEFAULTS: Record<string, RotSpinDefaults> = {
    torus:   {rotX: -90, spinY: 0.05},
    box:     {rotX: 15, spinY: 0.1},
    sphere:  {spinY: 0.1},
    capsule: {spinY: 0.1},
    mobius:  {rotX: -30, spinY: 0.05},
}

export interface ComponentProps {
    shape3d: string | Record<string, unknown>
    shape3dType: string
    center: Parameters<typeof transformPosition>[0]
    zoom: number
    glossiness: number
    lighting: number
    uvMode: string
    speed: number
}

export const componentDefinition: GpuShaderDefinition<ComponentProps> = defineShader<ComponentProps>({
    name: "Form3D",
    role: 'shapeEffect',
    species: 'custom',
    boundingBoxDeclaration: { aspectRatio: null },
    category: "Distortions",
    description: "Wraps child content onto a 3D raymarched shape with lighting.",
    requiresRTT: true,
    requiresChild: true,
    animatedTime: {speed: 'speed'},
    // The packed shape params the trace threads to the SDF each march step (rotation cos/sin + spin
    // accumulation + scaled sizes), driven CPU-side each frame from the shape JSON.
    extraFields: sdf3d.packedParamExtraFields('_f3p', [[1, 0, 1, 0], [1, 0, 0.35, 0.1], [0.3, 0.02, 0, 0]]),
    props: {
        shape3d: {
            default: DEFAULT_RIBBON,
            description: '3D shape and its parameters',
            ui: {type: 'shape3d', label: 'Shape', group: 'Shape'}
        },
        shape3dType: {
            default: 'ribbon',
            compileTime: true,
            description: 'Active shape type — triggers recompile when shape is switched',
        },
        center: {
            default: {x: 0.5, y: 0.5},
            transform: transformPosition,
            description: 'Center position of the shape on screen',
            ui: {type: 'position', label: 'Center', group: 'Position'}
        },
        zoom: {
            default: 50,
            description: 'Camera zoom level',
            ui: {type: ['range', 'map'], min: 10, max: 200, step: 1, label: 'Zoom', group: 'Scene'}
        },
        glossiness: {
            default: 50,
            description: 'Specular highlight intensity and sharpness',
            ui: {type: ['range', 'map'], min: 0, max: 200, step: 1, label: 'Glossiness', group: 'Lighting'}
        },
        lighting: {
            default: 50,
            description: 'Overall intensity of lighting effects',
            ui: {type: ['range', 'map'], min: 0, max: 200, step: 1, label: 'Lighting', group: 'Lighting'}
        },
        uvMode: {
            default: 'stretch',
            transform: transformEdges,
            description: 'How to handle UV coordinates at shape boundaries',
            ui: {
                type: 'select',
                options: [
                    {label: 'Stretch', value: 'stretch'},
                    {label: 'Mirror',  value: 'mirror'},
                    {label: 'Wrap',    value: 'wrap'},
                ],
                label: 'UV Edges',
                group: 'Scene'
            }
        },
        speed: {
            default: 1,
            description: 'Animation speed — scales all spin rates',
            ui: {type: 'range', min: -10, max: 10, step: 0.1, label: 'Speed', group: 'Animation'}
        },
    },

    gpu: {fragment: (params: GpuFragmentParams): Expr => {
        const {childNode, convertToTexture, uniforms, propValues, getCpuValue, onBeforeRender, ctx} = params
        if (!childNode) return ZERO

        const childTexture = convertToTexture(childNode)
        const shapeType = (propValues.shape3dType as string) || 'ribbon'
        const shape = SHAPES[shapeType] ?? SHAPES.ribbon

        // ── CPU driver: parse shape JSON → pack rotation cos/sin (+ spin accum) + scaled sizes ──
        let spinX = 0, spinY = 0, spinZ = 0
        let lastJson = ''
        let cfg: Record<string, number> = (() => { try { return JSON.parse(getCpuValue('shape3d') as string) } catch { return {} } })()
        const num = (k: string, f: number): number => (typeof cfg[k] === 'number' ? cfg[k] : f)
        const drive = (deltaTime: number) => {
            const raw = getCpuValue('shape3d')
            if (typeof raw === 'string') { if (raw !== lastJson) { lastJson = raw; try { cfg = JSON.parse(raw) } catch { /* keep */ } } }
            else if (raw && typeof raw === 'object') cfg = raw as Record<string, number>
            const speed = (getCpuValue('speed') as number) ?? 1

            if (shapeType === 'ribbon') {
                const angle = num('angle', 0) * DEG
                const twist = num('twist', 50) * 0.03
                const width = num('width', 40) * 0.012
                const halfW = width * 0.5
                const thickness = num('thickness', 20) * 0.001 + 0.01
                const seed = num('seed', 0)
                const lipschitz = Math.max(halfW * twist, 1)
                params.setExtraField('_f3p0', [angle, twist, halfW, width])
                params.setExtraField('_f3p1', [thickness, seed, lipschitz, 0])
                params.setExtraField('_f3p2', [0, 0, 0, 0])
                return
            }

            const dfl = SPIN_DEFAULTS[shapeType] ?? {}
            spinX += deltaTime * speed * num('spinX', dfl.spinX ?? 0) * TAU_RATE
            spinY += deltaTime * speed * num('spinY', dfl.spinY ?? 0) * TAU_RATE
            spinZ += deltaTime * speed * num('spinZ', dfl.spinZ ?? 0) * TAU_RATE
            const aX = num('rotX', dfl.rotX ?? 0) * DEG + spinX
            const aY = num('rotY', dfl.rotY ?? 0) * DEG + spinY
            const aZ = num('rotZ', dfl.rotZ ?? 0) * DEG + spinZ
            const rot0: number[] = [Math.cos(aX), Math.sin(aX), Math.cos(aY), Math.sin(aY)]
            const cz = Math.cos(aZ), sz = Math.sin(aZ)

            if (shapeType === 'sphere') {
                params.setExtraField('_f3p0', rot0)
                params.setExtraField('_f3p1', [cz, sz, num('radius', 60) * 0.008 + 0.1, 0])
                params.setExtraField('_f3p2', [0, 0, 0, 0])
            } else if (shapeType === 'torus') {
                params.setExtraField('_f3p0', rot0)
                params.setExtraField('_f3p1', [cz, sz, num('outerRadius', 60) * 0.008 + 0.1, num('tubeRadius', 25) * 0.006 + 0.05])
                params.setExtraField('_f3p2', [0, 0, 0, 0])
            } else if (shapeType === 'box') {
                params.setExtraField('_f3p0', rot0)
                params.setExtraField('_f3p1', [cz, sz, num('sizeX', 50) * 0.006, num('sizeY', 50) * 0.006])
                params.setExtraField('_f3p2', [num('sizeZ', 50) * 0.006, num('rounding', 20) * 0.001, 0, 0])
            } else if (shapeType === 'capsule') {
                params.setExtraField('_f3p0', rot0)
                params.setExtraField('_f3p1', [cz, sz, num('radius', 25) * 0.006 + 0.05, num('height', 80) * 0.005])
                params.setExtraField('_f3p2', [0, 0, 0, 0])
            } else { // mobius
                params.setExtraField('_f3p0', rot0)
                params.setExtraField('_f3p1', [cz, sz, num('ringRadius', 60) * 0.008 + 0.1, num('halfWidth', 30) * 0.005])
                params.setExtraField('_f3p2', [num('thickness', 8) * 0.001 + 0.005, 0, 0, 0])
            }
        }
        drive(0)
        onBeforeRender((fp: {deltaTime?: number}) => drive(fp?.deltaTime ?? 0))

        // ── March + surface UV + lighting: a visible staged composition — the kit's
        // param-threaded sphere-march (camera + normal live inside it), the per-shape surfaceUV
        // atlas (kit sdf3d primitives), kit applyUVMode3d edge routing, and kit distortion3dLighting.
        // The march core itself is ATOMIC (irreducible): a serial sphere-march whose loop-carried
        // ray parameter feeds the SDF each step — no separable stage exists without threading the
        // loop state through part boundaries. ──
        const t = animatedTime(params)
        const trace = buildParamThreadedTrace(shape.sdf, {steps: shape.steps, stepCap: shape.stepCap, hitEps: shape.hitEps})
        const traceRes = call(trace, `form3dTrace_${shapeType}`, [ctx.uv, ctx.aspect, uniforms.center, uniforms.zoom, uniforms._f3p0, uniforms._f3p1, uniforms._f3p2, t])
        const hitPos = traceRes.member('hitPos')
        const normal = traceRes.member('normal')
        const rd = traceRes.member('rd')
        const hit = traceRes.member('hit')

        const surfUV = call(shape.surfaceUV, shape.uvHint, [hitPos, normal, uniforms._f3p0, uniforms._f3p1, uniforms._f3p2, t])
        const uvm = call(applyUVMode3d, 'applyUVMode3d', [surfUV, uniforms.uvMode])
        const finalUV = uvm.member('xy')
        const uvVisible = uvm.member('z')
        // RTT is premultiplied → straight before lighting (same as the kit builder).
        const surfaceColor = call(blend.unpremultiplyAlpha, 'unpremultiplyAlpha', [childTexture.sample(finalUV)])
        const glossAmt = uniforms.glossiness.mul(0.005)
        const lightAmt = uniforms.lighting.mul(0.005)
        return call(distortion3dLighting, 'distortion3dLighting', [surfaceColor, normal, rd, hit, uvVisible, glossAmt, lightAmt])
    }}
})

export default componentDefinition
