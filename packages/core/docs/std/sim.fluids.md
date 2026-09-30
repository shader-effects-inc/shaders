# Fluids

Smoke, ink and fog that flow. A fluid is a square grid of velocity plus a **dye** (a density
and an age, or a color) that the engine advances with a full fluid solve every frame:
swirl, pressure, advection. `fluidSim` owns the state buffers, the output texture, the solve
chain, the pointer tracker, the clock and the idle gate that sleeps a settled field. You bring
the parts that make the look and declare them by role.

**Emitters** put dye and momentum in: `splat` at a fixed source (a cone from a point),
`cursorRibbon` along the pointer's stroke as a continuous ribbon of stamps. **Solve stages**
wrap every solve: `ambientForce` before it (a turbulence field), `restoreToward` after it (a
color the solve would otherwise blur away). **Init** decides how the field starts:
`seededFieldInit` seeds it and runs silent warm-up solves so the first visible frame already
flows. The `values` function turns props, the clock and the pointer into the numbers every pass
reads; positions and radii in it are in grid cells.

The fragment samples the output texture under the key you named and colors it however you
like. When an effect only needs a wave field or a displacement, `simulate.grid` says it with no
GPU code.

## Reach for it when

| When | Use |
|---|---|
| smoke rising from a point, stirred by the cursor | `fluidSim` with `splat` |
| smoke or ink painted along the pointer's stroke | `fluidSim` with `cursorRibbon` |
| fog that is already everywhere when the layer appears | `seededFieldInit` |
| a wind or turbulence over the whole field | `ambientForce` |
| a color that must not wash out over time | `restoreToward` |
| the simulation should sleep once the last stroke has faded | the fadeSeconds option of `cursorRibbon` |

## Order

- fluidSim
- splat
- cursorRibbon
- seededFieldInit
- ambientForce
- restoreToward

## Example

```ts
import {defineShader, p, sim, transformColor, transformPosition} from 'shaders/std'
import {tgpu, d} from '@coreroot/gpu/kit'
import {buildStableFluidsKernels, buildEmitterSplatKernel, buildFluidOutputKernel, mixExpr, vec4, ZERO} from '@coreroot/gpu/porters'
import type {GpuFragmentParams, KitTexture, Expr} from '@coreroot/gpu/porters'

const {fluidSim, splat} = sim.fluids

const N = 256
const COUNT = N * N

// The values every pass reads each frame. dt, curlStrength, velFade, dyeFade and colorDecay are the
// solver's; the rest drive the emitter and the cursor shove.
const FluidParams = d.struct({
  dt: d.f32, emitX: d.f32, emitY: d.f32, emitVelX: d.f32, emitVelY: d.f32,
  perpDirX: d.f32, perpDirY: d.f32, spreadFactor: d.f32, emitRad: d.f32, emitIntensity: d.f32,
  dyeFade: d.f32, velFade: d.f32, curlStrength: d.f32, gravity: d.f32, cursorX: d.f32, cursorY: d.f32,
  cursorVelX: d.f32, cursorVelY: d.f32, mouseActive: d.f32, mouseRadSq: d.f32, colorDecay: d.f32,
})
// The fluid's buffers, by the names the solver expects.
const fluidLayout = tgpu.bindGroupLayout({
  velA: {storage: d.arrayOf(d.vec4f, COUNT), access: 'mutable'},
  velB: {storage: d.arrayOf(d.vec4f, COUNT), access: 'mutable'},
  dyeA: {storage: d.arrayOf(d.vec4f, COUNT), access: 'mutable'},
  dyeB: {storage: d.arrayOf(d.vec4f, COUNT), access: 'mutable'},
  pressure: {storage: d.arrayOf(d.f32, COUNT), access: 'mutable'},
  divergence: {storage: d.arrayOf(d.f32, COUNT), access: 'mutable'},
  params: {uniform: FluidParams},
})
const outputLayout = tgpu.bindGroupLayout({
  dyeA: {storage: d.arrayOf(d.vec4f, COUNT), access: 'readonly'},
  outTex: {storageTexture: d.textureStorage2d('rgba16float', 'write-only')},
})

// The one pass that is this shader's own: a cone emitter at a fixed source, plus the cursor shove.
const splatKernel = buildEmitterSplatKernel(fluidLayout, {n: N, namePrefix: 'smoke', cone: true, cursorPush: true, densityGain: 8, velocityBlendGain: 3})
// Everything else is the shared solve over a density-and-age dye.
const solverKernels = buildStableFluidsKernels(fluidLayout, {n: N, namePrefix: 'smoke', boundary: 'clamped', dye: 'densityAge', dyeDissipation: true, ageAdvance: true})
const outputKernel = buildFluidOutputKernel(outputLayout, {n: N, namePrefix: 'smoke', dye: 'densityAge'})

// Smoke: fresh smoke in colorA ages toward colorB as it rises and dissipates.
export const Smoke = defineShader({
  name: 'Smoke',
  usesPointer: true,
  props: {
    colorA: {default: '#fc83f9', transform: transformColor},
    colorB: {default: '#c21c79', transform: transformColor},
    emitFrom: {default: {x: 0.5, y: 1}, transform: transformPosition},
    speed: {default: 20},
    emitRadius: {default: 0.08},
    dissipation: {default: 0.2},
    detail: {default: 25},
    mouseInfluence: {default: 0.1},
  },
  ...fluidSim(() => ({
    resolution: N,
    layout: fluidLayout, outputLayout, paramsSchema: FluidParams,
    solver: {kernels: solverKernels, jacobiIters: 10},
    output: {kernel: outputKernel, key: 'smokeTexture'},
    pointer: {minDrag: 0.0005},
    inject: [splat({kernel: splatKernel})],
    values: (f) => {
      const emit = f.getCpuValue('emitFrom') as {x: number; y: number} | undefined
      const velMag = f.num('speed', 20) * N * 0.15
      const ptr = f.ptr!
      const mouseInf = f.num('mouseInfluence', 0.1)
      const active = ptr.moving && mouseInf > 0
      const mouseRadGrid = 0.1 * N
      return {
        dt: f.dt,
        emitX: (emit?.x ?? 0.5) * N,
        emitY: (1 - (emit?.y ?? 0)) * N,
        emitVelX: 0, emitVelY: -velMag,
        perpDirX: 1, perpDirY: 0,
        spreadFactor: Math.tan(Math.PI / 6),
        emitRad: f.num('emitRadius', 0.08) * N,
        emitIntensity: 1,
        dyeFade: f.num('dissipation', 0.2),
        velFade: 0.2,
        curlStrength: f.num('detail', 25),
        gravity: 0.5,
        colorDecay: 0.4,
        cursorX: active ? ptr.x * N : 0,
        cursorY: active ? ptr.y * N : 0,
        cursorVelX: active ? ptr.dx * N * 15 * mouseInf : 0,
        cursorVelY: active ? ptr.dy * N * 15 * mouseInf : 0,
        mouseActive: active ? 1 : 0,
        mouseRadSq: mouseRadGrid * mouseRadGrid,
      }
    },
  })),
  gpu: {
    fragment: ({ctx, computeOutputs, uniforms}: GpuFragmentParams): Expr => {
      const field = computeOutputs?.smokeTexture as KitTexture | undefined
      if (!field) return ZERO // no GPU: transparent
      const s = field.sample(ctx.uv, 'linearClamp') // x = density, y = age
      const color = mixExpr(uniforms.colorA, uniforms.colorB, s.member('y'))
      return vec4(color.member('rgb'), color.member('a').mul(s.member('x')))
    },
  },
})
```
