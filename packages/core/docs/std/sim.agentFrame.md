# Agent frame

The once-per-frame arithmetic an agent simulation writes into its values. Inside `agentSim`'s
`frame` callback you turn props and the pointer into the numbers the GPU steps read. These are
the recipes that come up in every agent layer, as plain functions with no GPU in them: a drag
factor that does not depend on frame rate (`expDecay`), the pointer in the shape's own
coordinates (`pointerToShapeLocal`), a grid that fits the agent count (`fitJitteredGrid`,
`fitIsoGrid`), a direction that follows the cursor's travel (`createMotionAxis`), camera rows
for a 3D view (`cameraRowsYDown`), and the rotation bookkeeping a spinning shape needs to
carry its swarm along (`omegaFromRotationDeltas`, `entrainmentFromOmega`, `r3SubCellOffset`).

Call them each frame, put the results in the values struct with `sys.writeParams`, and return
`sys.frame({count})`. Anything stateful (`createMotionAxis`) is created once, outside the
per-frame function, so each layer instance keeps its own.

## Reach for it when

| When | Use |
|---|---|
| a velocity or gust that fades at a rate per second | `expDecay` |
| the cursor as a force inside a centered, scaled, rotated shape | `pointerToShapeLocal` |
| the magnet's axis should follow the cursor's motion | `createMotionAxis` |
| agents resting on a grid that fits the count and the aspect | `fitJitteredGrid` with `field.jitteredGridHome` |
| a W×H grid of particles over the canvas | `fitIsoGrid` |
| a 3D view of the agents from camera angles | `cameraRowsYDown` |
| a swarm carried along by its rotating shape | `omegaFromRotationDeltas` then `entrainmentFromOmega` |
| the density grid of `force.pressure` should not line up with the shape | `r3SubCellOffset` |

## Order

- expDecay
- pointerToShapeLocal
- createMotionAxis
- fitJitteredGrid
- fitIsoGrid
- cameraRowsYDown
- omegaFromRotationDeltas
- entrainmentFromOmega
- r3SubCellOffset

## Example

```ts
import {defineShader, sim, transformColor} from 'shaders/std'
import {tgpu, d, agents} from '@coreroot/gpu/kit'
import {makeCpuValueGetter, readAgentFrame, resolveRenderRes, SIZE_REF_RES} from '@coreroot/gpu/porters'

const {agentSim, renderAgents, force, drift, integrator, agentFrame, pointSlot} = sim.agents

const MAX_MOTES = 8192
const RES = resolveRenderRes({desktop: 1024, mobile: 512})
const DEG_TO_RAD = Math.PI / 180

const SimParams = d.struct({
  color: d.vec4f,
  count: d.f32, dt: d.f32, time: d.f32, aspect: d.f32, domainX: d.f32,
  driftBase: d.f32, angleRad: d.f32, speedVar: d.f32, angleVarRad: d.f32,
  randomness: d.f32, twinkle: d.f32, softness: d.f32,
  bodyR: d.f32,
  cursorX: d.f32, cursorY: d.f32, cursorRadSq: d.f32, cursorForce: d.f32, dragMul: d.f32,
})
const simLayout = tgpu.bindGroupLayout({
  agents: {storage: d.arrayOf(d.vec4f, MAX_MOTES), access: 'mutable'},
  accumE: {storage: d.arrayOf(d.atomic(d.u32), RES * RES), access: 'mutable'},
  params: {uniform: SimParams},
  outTex: {storageTexture: d.textureStorage2d('rgba16float', 'write-only')},
})

const motesInit = agents.makeUniformScatterInit(simLayout, 'motesInit')
const motesUpdate = integrator.drift2d(simLayout, {drift: drift.variedHeading(simLayout), gust: force.cursorGust(simLayout), name: 'motesUpdate'})
const motesRender = renderAgents.pointWorld(simLayout, {
  shape: 'dot', res: RES,
  place: pointSlot.orbitalPlace(simLayout, {radius: 0.035, rate: 1.4}),
  brightness: pointSlot.twinkleBrightness(simLayout, {freq: 2}),
  bodyRadius: pointSlot.presenceRadius(simLayout),
  names: {splat: 'motesSplat', resolve: 'motesResolve'},
})

// Motes: drifting particles the cursor can stir. The frame turns props into the values the steps
// read; the gust's fade is a rate per second, made frame-rate independent by expDecay.
export const Motes = defineShader({
  name: 'Motes',
  usesPointer: true,
  props: {
    particleColor: {default: '#ffffff', transform: transformColor},
    count: {default: 1200},
    speed: {default: 0.25},
    angle: {default: 90},
    particleSize: {default: 1.2},
    cursorStrength: {default: 0.5},
  },
  ...agentSim<typeof SimParams>({
    layout: simLayout,
    params: SimParams,
    maxAgents: MAX_MOTES,
    countCap: {desktop: MAX_MOTES, mobile: 3000},
    output: {key: 'outTex', name: 'particleTexture', size: [RES, RES], format: 'rgba16float'},
    bake: () => ({
      pipelines: {
        init: {kernel: motesInit, threads: 'max'},
        update: {kernel: motesUpdate, threads: 'agents'},
        splat: motesRender.splat,
        resolve: motesRender.resolve,
      },
      initStep: 'init',
      program: ['update', 'splat', 'resolve'],
    }),
    frame: (sys, {getCpuValue}) => {
      const g = makeCpuValueGetter(getCpuValue)
      let localTime = 0
      return (frameParams) => {
        const {dt, aspect, pointerX, pointerY} = readAgentFrame(frameParams)
        localTime += dt
        const count = sys.resolveCount(g('count', 1200))
        const col = getCpuValue('particleColor') as {x: number; y: number; z: number; w: number} | undefined
        sys.writeParams({
          color: d.vec4f(col?.x ?? 1, col?.y ?? 1, col?.z ?? 1, col?.w ?? 1),
          count, dt, time: localTime, aspect, domainX: Math.max(aspect, 0.01),
          driftBase: Math.min(Math.max(g('speed', 0.25), 0), 1.5) * 0.48,
          angleRad: g('angle', 90) * DEG_TO_RAD + Math.PI,
          speedVar: 0.3,
          angleVarRad: 30 * DEG_TO_RAD,
          randomness: 0.25,
          twinkle: 0.5,
          softness: 0.1,
          bodyR: Math.min(Math.max(g('particleSize', 1.2), 0.3), 4) * 2 / SIZE_REF_RES,
          cursorX: pointerX * aspect, cursorY: pointerY,
          cursorRadSq: 0.22 * 0.22,
          cursorForce: Math.min(Math.max(g('cursorStrength', 0.5), 0), 1) * 3.5,
          // The gust fades at 1.1 per second, whatever the frame rate.
          dragMul: agentFrame.expDecay(1.1, dt),
        })
        return sys.frame({count})
      }
    },
    fragment: {output: 'particleTexture', fallback: 'transparent'},
  }),
})
```
