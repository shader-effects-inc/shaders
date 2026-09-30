# Agent rendering

How agents become a picture. Every agent frame ends the same way: each agent stamps its shape
into a shared canvas (the **splat**), then one pass turns the canvas into the output picture
and clears it for the next frame (the **resolve**). `renderAgents` in `sim.agents` pairs a
splat with its resolve for you; reach into this module when you need another pairing, or for
the `pointSlot` parts that give round motes their placement, brightness and size.

Four splats match the four kinds of agent: `splat.orientedWorld` draws a shape along the
agent's heading in canvas units; `splat.pointWorld` draws round motes with a live softness;
`splat.volume` draws a 3D swarm in perspective; `splat.relief` draws particles standing on an
image through a camera. Three resolves close them: `resolve.ramp` colors from a rest color to
an excited color, with optional trails; `resolve.tint` gives one color an alpha from coverage;
`resolve.weightedColor` averages the colors that landed, nearest first. A splat dispatches over
the agents; a resolve dispatches over every texel of the canvas, because it also clears it.

## Reach for it when

| When | Use |
|---|---|
| a splat and resolve paired for you | `renderAgents` |
| motes that wander in small circles, twinkle, and vary in size | `pointSlot.orbitalPlace`, `pointSlot.twinkleBrightness`, `pointSlot.presenceRadius` |
| arrows, streaks or comets along the heading | `splat.orientedWorld` with `resolve.ramp` |
| round motes in one color | `splat.pointWorld` with `resolve.tint` |
| a 3D swarm, nearer particles larger and brighter | `splat.volume` with `resolve.ramp` |
| particles carrying the child's color, nearer ones in front | `splat.relief` with `resolve.weightedColor` |
| motion trails behind the agents | the trails option of `resolve.ramp` |

## Order

- pointSlot
- pointSlot.orbitalPlace
- pointSlot.twinkleBrightness
- pointSlot.presenceRadius
- splat
- splat.orientedWorld
- splat.pointWorld
- splat.volume
- splat.relief
- resolve
- resolve.ramp
- resolve.tint
- resolve.weightedColor

## Example

```ts
import {defineShader, sim, transformColor} from 'shaders/std'
import {tgpu, d, agents} from '@coreroot/gpu/kit'
import {makeCpuValueGetter, readAgentFrame, resolveRenderRes, SIZE_REF_RES} from '@coreroot/gpu/porters'

const {agentSim, force, drift, integrator, agentFrame} = sim.agents
const {splat, resolve, pointSlot} = sim.agentRender

const MAX_MOTES = 8192
const RES = resolveRenderRes({desktop: 1024, mobile: 512})

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

// The look, paired by hand: a point splat whose placement, brightness and size are pointSlot
// parts, closed by the tint resolve over the same canvas.
const motesSplat = splat.pointWorld(simLayout, {
  shape: 'glow', res: RES,
  place: pointSlot.orbitalPlace(simLayout, {radius: 0.035, rate: 1.4}),
  brightness: pointSlot.twinkleBrightness(simLayout, {freq: 2}),
  bodyRadius: pointSlot.presenceRadius(simLayout),
  name: 'motesSplat',
})
const motesResolve = resolve.tint(simLayout, {res: RES, name: 'motesResolve'})

// Motes: soft drifting points that twinkle and circle lazily around their path.
export const Motes = defineShader({
  name: 'Motes',
  usesPointer: true,
  props: {
    particleColor: {default: '#ffffff', transform: transformColor},
    count: {default: 1200},
    speed: {default: 0.25},
    randomness: {default: 0.25},
    twinkle: {default: 0.5},
    softness: {default: 0.1},
    particleSize: {default: 1.2},
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
        splat: {kernel: motesSplat, threads: 'agents'},
        resolve: {kernel: motesResolve, threads: 'fixed', size: [RES, RES]},
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
          angleRad: Math.PI * 1.5,
          speedVar: 0.3,
          angleVarRad: Math.PI / 6,
          randomness: Math.min(Math.max(g('randomness', 0.25), 0), 1),
          twinkle: Math.min(Math.max(g('twinkle', 0.5), 0), 1),
          softness: Math.min(Math.max(g('softness', 0.1), 0), 1),
          bodyR: Math.min(Math.max(g('particleSize', 1.2), 0.3), 4) * 2 / SIZE_REF_RES,
          cursorX: pointerX * aspect, cursorY: pointerY,
          cursorRadSq: 0.22 * 0.22,
          cursorForce: 0,
          dragMul: agentFrame.expDecay(1.1, dt),
        })
        return sys.frame({count})
      }
    },
    fragment: {output: 'particleTexture', fallback: 'transparent'},
  }),
})
```
