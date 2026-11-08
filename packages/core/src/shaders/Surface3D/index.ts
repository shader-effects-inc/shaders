import type {GpuShaderDefinition, GpuFragmentParams, KitTexture, Expr} from "@coreroot/gpu/porters"
import {
    call, ZERO, createPingPongPair, type ComputeStep,
    makeWaveEquationKernel, makeHeightFieldMarchKernel, createHeightFieldParamsUniform,
    createWaveStateBuffers, createKernelPass, waveEquationLayout, heightFieldMarchLayout,
    WAVE_GRID, WAVE_HALFEXTENT, CURSOR_WAVE_BOUND, WAVE_DECAY, WAVE_SETTLE_MS,
    HEIGHT_SLAB_PAD, HEIGHT_FIELD_STATE_FORMAT,
} from "@coreroot/gpu/porters"
import {defineShader, recompileWhen} from "@coreroot/std"
import {clamp, local, mul, vec3, vec4} from "@coreroot/std/math"
import {blend, constants} from "@coreroot/gpu/kit"
import {transformColor, transformEdges} from "@coreroot/utilities/transformations"
import {isMobileGpuViewport} from "@coreroot/utilities/device"

// Raymarch grid cap. This is the dominant per-frame cost (one full march per texel), so it's
// device-tiered in the same style as sdf3d's VOLUMETRIC_FIELD_RES/_MOBILE split, then aspect-fitted
// to the real canvas at composition time (see `compute`). Build-time constant (SSR/tests → desktop).
const COMPUTE_MAX = isMobileGpuViewport() ? 960 : 1600
const COMPUTE_FALLBACK_W = 1600
const COMPUTE_FALLBACK_H = 900
// Fixed march steps across the height slab (the scaffold's slab clamp) + bisection refinements.
const MARCH_STEPS = isMobileGpuViewport() ? 12 : 16
const MARCH_REFINE = 6
const DEG_TO_RAD = constants.DEG_TO_RAD

export interface ComponentProps {
    amplitude: number
    waveType: string
    frequency: number
    octaves: number
    seed: number
    speed: number
    tilt: number
    roll: number
    height: number
    zoom: number
    nearCutoff: number
    farCutoff: number
    edgePinning: number
    edges: string
    lighting: number
    glossiness: number
    highlights: number
    lightX: number
    lightY: number
    lightZ: number
    lightColor: Parameters<typeof transformColor>[0]
    cursorIntensity: number
    cursorSpeed: number
}

export const componentDefinition: GpuShaderDefinition<ComponentProps> = defineShader<ComponentProps>({
    name: "Surface3D",
    role: 'shapeEffect',
    species: 'custom',
    boundingBoxDeclaration: { aspectRatio: null },
    category: "Distortions",
    description: "Drapes child content over a 3D wave surface with perspective and lighting",
    requiresRTT: true,
    requiresChild: true,
    usesPointer: true,
    providesUVContextViaCompute: true,
    props: {
        amplitude: { default: 0.3, description: "Wave height", ui: { type: 'range', min: 0, max: 1, step: 0.01, label: 'Amplitude', group: 'Waves' } },
        waveType: { default: 'fractal', compileTime: true, description: "Wave pattern type", ui: { type: 'select', options: [{label: 'Fractal', value: 'fractal'}, {label: 'Sine', value: 'sine'}, {label: 'Ridge', value: 'ridge'}], label: 'Wave Type', group: 'Waves' } },
        frequency: { default: 1.5, description: "Wave frequency", ui: { type: 'range', min: 0.1, max: 5, step: 0.1, label: 'Frequency', group: 'Waves' } },
        // Only the clamped integer reaches the kernel, so a sub-integer drag position must NOT
        // recompose — a recompose here destroys/reallocates the compute textures and visibly resets
        // the cursor-ripple field to zero (GridDistortion.gridSize pattern).
        octaves: { default: 2, recompile: recompileWhen((prev: number, next: number) => Math.max(1, Math.floor(prev)) !== Math.max(1, Math.floor(next))), description: "Detail octaves", ui: { type: 'range', min: 1, max: 5, step: 1, label: 'Octaves', group: 'Waves' } },
        seed: { default: 0, description: "Random seed offset", ui: { type: 'range', min: 0, max: 100, step: 1, label: 'Seed', group: 'Waves' } },
        speed: { default: 0.5, description: "Animation speed", ui: { type: 'range', min: 0, max: 3, step: 0.1, label: 'Speed', group: 'Waves' } },
        tilt: { default: 35, description: "Camera tilt in degrees", ui: { type: 'range', min: 0, max: 85, step: 1, label: 'Tilt', group: 'Camera' } },
        roll: { default: 0, description: "Camera roll in degrees", ui: { type: 'range', min: -45, max: 45, step: 1, label: 'Roll', group: 'Camera' } },
        height: { default: 0, description: "Camera height", ui: { type: 'range', min: -1, max: 2, step: 0.01, label: 'Height', group: 'Camera' } },
        zoom: { default: 1, description: "Camera zoom", ui: { type: 'range', min: 0.3, max: 3, step: 0.01, label: 'Zoom', group: 'Camera' } },
        nearCutoff: { default: 0, description: "Near fade distance", ui: { type: 'range', min: 0, max: 1, step: 0.01, label: 'Near Cutoff', group: 'Camera' } },
        farCutoff: { default: 1, description: "Far fade distance", ui: { type: 'range', min: 0, max: 1, step: 0.01, label: 'Far Cutoff', group: 'Camera' } },
        edgePinning: { default: 0, description: "Smoothly flatten the surface at its edges (orthogonal to edge mode)", ui: { type: 'range', min: 0, max: 1, step: 0.01, label: 'Edge Pinning', group: 'Waves' } },
        edges: { default: 'mirror', transform: transformEdges, compileTime: true, description: "How to handle content beyond the surface bounds", ui: { type: 'select', options: [{label: 'Stretch', value: 'stretch'}, {label: 'Transparent', value: 'transparent'}, {label: 'Mirror', value: 'mirror'}, {label: 'Wrap', value: 'wrap'}], label: 'Edges', group: 'Waves' } },
        lighting: { default: 30, recompile: recompileWhen((prev: number, next: number) => (prev < 0.01) !== (next < 0.01)), description: "Intensity of lighting and shading", ui: { type: 'range', min: 0, max: 100, step: 1, label: 'Lighting', group: 'Lighting' } },
        glossiness: { default: 0, description: "Surface glossiness", ui: { type: 'range', min: 0, max: 100, step: 1, label: 'Glossiness', group: 'Lighting' } },
        highlights: { default: 15, description: "Specular highlight intensity", ui: { type: 'range', min: 0, max: 100, step: 1, label: 'Highlights', group: 'Lighting' } },
        lightX: { default: 0.4, description: "Light direction X", ui: { type: 'range', min: -1, max: 1, step: 0.01, label: 'Light X', group: 'Lighting' } },
        lightY: { default: -0.6, description: "Light direction Y", ui: { type: 'range', min: -1, max: 1, step: 0.01, label: 'Light Y', group: 'Lighting' } },
        lightZ: { default: 0.7, description: "Light direction Z", ui: { type: 'range', min: -1, max: 1, step: 0.01, label: 'Light Z', group: 'Lighting' } },
        lightColor: { default: '#ffffff', transform: transformColor, description: "Light color", ui: { type: 'color', label: 'Light Color', group: 'Lighting' } },
        cursorIntensity: { default: 1, recompile: recompileWhen((prev: number, next: number) => (prev < 0.01) !== (next < 0.01)), description: "Strength of cursor ripples", ui: { type: 'range', min: 0, max: 2, step: 0.01, label: 'Cursor Ripples', group: 'Interaction' } },
        cursorSpeed: { default: 0.5, description: "Speed of cursor ripple propagation", ui: { type: 'range', min: 0, max: 1, step: 0.01, label: 'Ripple Speed', group: 'Interaction' } }
    },

    // WebGPU compute: raymarch a wavy 3D surface into two fixed-res maps — uvMaskMap (hitU, hitV,
    // hit×mask) + litMap (surface lighting) — via the kit's height-field surface machine
    // (scaffolds/heightField: spectral heightfield + wave-equation ripple layer + slab-clamped
    // march). The composer's UV-context path samples uvMaskMap at screenUV → the child generator's
    // uvContext (so the child rasterises once at the looked-up UV); else the fragment RTT-samples
    // the child at that UV. Cursor ripples run the scaffold's wave-equation on ping-pong state
    // buffers (cursorIntensity ≥ 0.01, compileTimeWhen-gated). The camera basis + cursor→wave-grid
    // projection are resolved on the CPU each frame.
    // `_upstreamUVMap` multi-provider chaining (sampling a parent provider's map inside this
    // compute) is unsupported — standalone uses the compute-pixel coord directly. Map-driven
    // props → no compute → transparent.
    compute: (params: GpuFragmentParams) => {
        const {gpu, getCpuValue, registerComputeTexture, onCleanup, onResize, dimensions} = params
        // NB: no childNode guard — the raymarch is child-INDEPENDENT (it produces the uvMaskMap/litMap
        // from the wave surface), and in the composer's UV-context path childNode is undefined (the
        // child generator is folded via uvContext, not passed here). The compute must still run there.
        const root = gpu?.root
        if (!root) return null // GPU-free resolve/tests: fragment falls back to ZERO.

        const waveType = (getCpuValue('waveType') as string) === 'sine' ? 1 : (getCpuValue('waveType') as string) === 'ridge' ? 2 : 0
        const octaveCount = Math.max(1, Math.floor((getCpuValue('octaves') as number) ?? 2))
        const edgeMode = (getCpuValue('edges') as number) ?? 2
        const lightingEnabled = ((getCpuValue('lighting') as number) ?? 30) >= 0.01
        const cursorEnabled = ((getCpuValue('cursorIntensity') as number) ?? 1) >= 0.01

        // Aspect-fit the raymarch grid to the canvas within the device cap, and never resolve ABOVE the
        // canvas — a 600px-wide canvas can't show a 1600-wide grid, and a portrait canvas got a
        // horizontally-over / vertically-under-resolved 16:9 grid. Sized once (no recompose on resize);
        // a degenerate first-frame canvas falls back to the cap at 16:9.
        const canvasW = Math.round(dimensions.width)
        const canvasH = Math.round(dimensions.height)
        let computeW = Math.min(COMPUTE_MAX, COMPUTE_FALLBACK_W)
        let computeH = Math.min(Math.round(COMPUTE_MAX * COMPUTE_FALLBACK_H / COMPUTE_FALLBACK_W), COMPUTE_FALLBACK_H)
        if (canvasW >= 16 && canvasH >= 16) {
            const fit = Math.min(1, COMPUTE_MAX / Math.max(canvasW, canvasH))
            computeW = Math.max(16, Math.round(canvasW * fit))
            computeH = Math.max(16, Math.round(canvasH * fit))
        }

        const uvMaskTex = root.createTexture({size: [computeW, computeH], format: HEIGHT_FIELD_STATE_FORMAT}).$usage('storage', 'sampled')
        const litTex = root.createTexture({size: [computeW, computeH], format: HEIGHT_FIELD_STATE_FORMAT}).$usage('storage', 'sampled')
        onCleanup(() => { uvMaskTex.destroy(); litTex.destroy() })
        const uvMaskMap = registerComputeTexture(uvMaskTex)
        const litMap = registerComputeTexture(litTex)

        // Wave state buffers (always allocated; only dispatched + read when cursorEnabled).
        const {bufferA, bufferB, readBuf} = createWaveStateBuffers(root)

        const paramsU = createHeightFieldParamsUniform(root)
        let curW = Math.max(1, Math.round(dimensions.width))
        let curH = Math.max(1, Math.round(dimensions.height))
        onResize(({width, height}) => { curW = Math.max(1, Math.round(width)); curH = Math.max(1, Math.round(height)) })

        const computeKernel = makeHeightFieldMarchKernel({
            waveType, octaveCount, edgeMode, lightingEnabled, cursorEnabled,
            marchSteps: MARCH_STEPS, marchRefine: MARCH_REFINE, computeW, computeH,
        })
        const computeBg = root.createBindGroup(heightFieldMarchLayout, {readBuf, params: paramsU.buffer as never, uvMaskTex, litTex})
        const computePass = createKernelPass(root, computeKernel, {size: [computeW, computeH], bindGroup: computeBg})

        const propagate = cursorEnabled ? makeWaveEquationKernel() : null
        // Ping-pong the ripple field: each step reads one buffer and writes the other, snapshotting
        // into `readBuf` for the raymarch. Both orientations are pre-built (nothing when the cursor
        // wave is compiled out).
        const wave = createPingPongPair(bufferA, bufferB)
        const propGroup = cursorEnabled
            ? wave.groups((readSrc, writeBuf) =>
                root.createBindGroup(waveEquationLayout, {readSrc, writeBuf, readBuf, params: paramsU.buffer as never}))
            : null
        const propagatePass = cursorEnabled && propagate
            ? createKernelPass(root, propagate, {size: [WAVE_GRID, WAVE_GRID]})
            : null

        let prevPx = 0.5
        let prevPy = 0.5
        let lastTime = Date.now()
        // Simulated seconds since the pointer last moved — not wall-clock, so a tab switch (rAF
        // stopped, zero frames stepped) cannot declare the wave field settled while it is still live.
        let simSinceActive = 0
        // Self-accumulated animated time (speed-scaled; speed=0 pauses).
        let animT = 0
        let lastSig: string | null = null

        // Returns a signature of everything that reaches the params struct, so `getComputeNodes` can
        // tell whether the raymarch would reproduce the previous frame verbatim.
        const writeParams = (fp: {pointer?: {x: number; y: number}; deltaTime?: number}, mouseSpeed: number, cursorActive: number): string => {
            const aspect = Math.max(1, curW) / Math.max(1, curH)
            const tilt = ((getCpuValue('tilt') as number) ?? 35) * DEG_TO_RAD
            const roll = ((getCpuValue('roll') as number) ?? 0) * DEG_TO_RAD
            const camHeight = (getCpuValue('height') as number) ?? 0
            const zoom = (getCpuValue('zoom') as number) ?? 1
            const {camPos, fwd, camRight, camUp, focal} = resolveOrbitCamera(tilt, roll, camHeight, zoom)

            // Cursor → z=0-plane world UV → wave-grid UV (height refinement dropped, w7b note).
            const px = fp.pointer?.x ?? 0.5
            const py = fp.pointer?.y ?? 0.5
            const cNdcX = (px - 0.5) * 2 * aspect
            const cNdcY = -(py - 0.5) * 2
            const rd = norm3([fwd[0] * focal + camRight[0] * cNdcX + camUp[0] * cNdcY, fwd[1] * focal + camRight[1] * cNdcX + camUp[1] * cNdcY, fwd[2] * focal + camRight[2] * cNdcX + camUp[2] * cNdcY])
            const cRayDownZ = Math.max(-rd[2], 0.001)
            const cT0 = camPos[2] / cRayDownZ
            const cWX0 = camPos[0] + rd[0] * cT0
            const cWY0 = camPos[1] + rd[1] * cT0
            const cursorWorldUVx = cWX0 * 0.5 + 0.5
            const cursorWorldUVy = cWY0 * 0.5 + 0.5
            const lc = getCpuValue('lightColor') as {x: number; y: number; z: number} | undefined

            const amp = (getCpuValue('amplitude') as number) ?? 0.3
            const cursorIntensity = (getCpuValue('cursorIntensity') as number) ?? 1
            const freq = (getCpuValue('frequency') as number) ?? 1.5
            const seed = (getCpuValue('seed') as number) ?? 0
            const edgePin = (getCpuValue('edgePinning') as number) ?? 0
            const lighting = ((getCpuValue('lighting') as number) ?? 30) * 0.035
            const glossiness = ((getCpuValue('glossiness') as number) ?? 0) * 0.01
            const highlights = ((getCpuValue('highlights') as number) ?? 15) * 0.04
            const lightX = (getCpuValue('lightX') as number) ?? 0.4
            const lightY = (getCpuValue('lightY') as number) ?? -0.6
            const lightZ = (getCpuValue('lightZ') as number) ?? 0.7
            const nearCutoff = (getCpuValue('nearCutoff') as number) ?? 0
            const farCutoff = (getCpuValue('farCutoff') as number) ?? 1
            const waveSpeed = (getCpuValue('cursorSpeed') as number) ?? 0.5
            const cursorWaveX = (cursorWorldUVx - 0.5) / WAVE_HALFEXTENT + 0.5
            const cursorWaveY = (cursorWorldUVy - 0.5) / WAVE_HALFEXTENT + 0.5
            // |z| the surface can reach: the base wave (|waveVal| ≤ 1, pinMask ≤ 1 → ≤ amp) plus the
            // live ripple contribution, which is dropped entirely once the field has settled so the
            // idle case gets the tightest possible march.
            const heightEnvelope = amp + HEIGHT_SLAB_PAD + (cursorActive > 0 ? cursorIntensity * 0.05 * CURSOR_WAVE_BOUND : 0)

            paramsU.write({
                amp, freq, seed, edgePin, cursorIntensity,
                t: animT,
                aspect, focal,
                camPos, camForward: fwd, camRight, camUp,
                lighting, glossiness, highlights, lightX, lightY, lightZ,
                lightColor: [lc?.x ?? 1, lc?.y ?? 1, lc?.z ?? 1],
                nearCutoff, farCutoff,
                cursorWaveX, cursorWaveY,
                mouseSpeed,
                dt: 0.016,
                decay: WAVE_DECAY,
                radius: 0.025,
                waveSpeed,
                heightEnvelope, cursorActive,
            })

            return [
                amp, freq, seed, edgePin, cursorIntensity, animT, aspect, tilt, roll, camHeight, zoom,
                lighting, glossiness, highlights, lightX, lightY, lightZ, lc?.x ?? 1, lc?.y ?? 1, lc?.z ?? 1,
                nearCutoff, farCutoff, cursorWaveX, cursorWaveY, mouseSpeed, waveSpeed, heightEnvelope, cursorActive,
            ].join(',')
        }

        return {
            outputs: {uvMaskMap, litMap},
            getComputeNodes: (frameParams: unknown): ComputeStep[] => {
                const fp = frameParams as {pointer?: {x: number; y: number}; deltaTime?: number}
                const now = Date.now()
                const dt = Math.min(fp.deltaTime ?? (now - lastTime) / 1000, 0.016)
                lastTime = now
                const px = fp.pointer?.x ?? prevPx
                const py = fp.pointer?.y ?? prevPy
                const rawVelX = dt > 0 ? (px - prevPx) / dt : 0
                const rawVelY = dt > 0 ? (py - prevPy) / dt : 0
                const mouseSpeed = Math.min(Math.sqrt(rawVelX * rawVelX + rawVelY * rawVelY), 2)
                prevPx = px
                prevPy = py
                if (mouseSpeed > 0.01) simSinceActive = 0
                else simSinceActive += dt
                animT += dt * ((getCpuValue('speed') as number) ?? 0.5) // speed=0 pauses
                // The wave field is at rest (or disabled) → it can no longer move the surface, so the
                // raymarch becomes a pure function of the params struct alone.
                const waveSettled = !cursorEnabled || !propagatePass || (simSinceActive * 1000 > WAVE_SETTLE_MS && mouseSpeed < 0.01)
                const sig = writeParams(fp, mouseSpeed, waveSettled ? 0 : 1)

                // Identical params + a settled field reproduce the previous frame exactly — skip the
                // whole computeW×computeH-thread raymarch, not just the wave propagation.
                if (waveSettled && sig === lastSig) return []
                lastSig = sig

                if (waveSettled || !propagatePass || !propGroup) return [computePass]
                const prop = propagatePass.with(propGroup())
                wave.swap()
                return [prop, computePass]
            },
        }
    },

    gpu: {fragment: (params: GpuFragmentParams): Expr => {
        const {childNode, ctx, computeOutputs, convertToTexture} = params
        if (!childNode) return ZERO
        const uvMaskTex = computeOutputs?.uvMaskMap as KitTexture | undefined
        const litTex = computeOutputs?.litMap as KitTexture | undefined
        if (!uvMaskTex || !litTex) return ZERO // GPU-free resolve / no device / map-driven → transparent.

        const uvMaskSample = local(uvMaskTex.sample(ctx.uv, 'linearClamp'), 'uvMaskSample')
        const litSample = litTex.sample(ctx.uv, 'linearClamp')
        const lookedUpUV = vec4(uvMaskSample.member('r'), uvMaskSample.member('g'), 0, 0).member('xy')
        const mask = uvMaskSample.member('b')
        const lit = litSample.member('rgb')

        // UV-propagation path: the composer pre-evaluated the child generator at the looked-up UV, so
        // `childNode` is already the color there (w7b: signalled via params.uvPropagationActive). RTT
        // fallback: child was composed at screenUV → re-sample it at the looked-up UV.
        const childColor = local(
            (params as {uvPropagationActive?: boolean}).uvPropagationActive
                ? childNode
                : convertToTexture(childNode).sample(lookedUpUV),
            'childColor',
        )
        // Composite the child color with the surface lighting + coverage mask (straight alpha out).
        const finalRgb = clamp(mul(childColor.member('xyz'), lit), vec3(0, 0, 0), vec3(1, 1, 1))
        const composed = vec4(finalRgb, mul(childColor.member('w'), mask))
        return call(blend.unpremultiplyAlpha, 'unpremultiplyAlpha', [composed])
    }}
})

// CPU vec3 helpers for the camera basis (kept local — the renderer resolves the camera per frame).
function cross3(a: [number, number, number], b: [number, number, number]): [number, number, number] {
    return [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]]
}
function norm3(a: [number, number, number]): [number, number, number] {
    const l = Math.max(1e-6, Math.hypot(a[0], a[1], a[2]))
    return [a[0] / l, a[1] / l, a[2] / l]
}

/**
 * The CAMERA stage: tilt/roll/height/zoom → an orbit camera position + orthonormal roll-rotated
 * basis + focal length, resolved on the CPU each frame (so the kernels do no per-thread camera
 * trig). Angles arrive in radians.
 */
function resolveOrbitCamera(tilt: number, roll: number, camHeight: number, zoom: number): {
    camPos: [number, number, number]
    fwd: [number, number, number]
    camRight: [number, number, number]
    camUp: [number, number, number]
    focal: number
} {
    const camDist = 0.85 / Math.max(0.0001, zoom)
    const camY = -Math.sin(tilt) * camDist
    const orbitZ = Math.cos(tilt) * camDist
    const camPos: [number, number, number] = [0, camY, orbitZ + camHeight]
    const fwd = norm3([0, -camY, -orbitZ])
    const right0: [number, number, number] = [1, 0, 0]
    const up0 = norm3(cross3(right0, fwd))
    const cr = Math.cos(roll)
    const sr = Math.sin(roll)
    const camRight: [number, number, number] = [right0[0] * cr + up0[0] * sr, right0[1] * cr + up0[1] * sr, right0[2] * cr + up0[2] * sr]
    const camUp: [number, number, number] = [up0[0] * cr - right0[0] * sr, up0[1] * cr - right0[1] * sr, up0[2] * cr - right0[2] * sr]
    return {camPos, fwd, camRight, camUp, focal: 1.5}
}

export default componentDefinition
