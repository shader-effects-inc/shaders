# Agents

Thousands of small things that move: flocks, motes, filings, swarms, particles standing on an
image. An agent simulation keeps a state per agent, moves every agent one step per frame, draws
each one into a shared canvas, and hands the result to the layer. `agentSim` owns the plumbing:
the state buffers, the output texture, the spawn on the first frame and again on a seed change,
the lower count on phones, the child's picture when one is needed, and the fragment that
samples the picture over the full canvas.

You declare three things. The **physics**: a list of `force` or `torque` parts folded into one
update step by an `integrator` (see `sim.agentForces`). The **look**: one `renderAgents`
variant that pairs a splat (each agent stamps its shape) with its resolve (the canvas becomes
the picture and is cleared for the next frame). The **frame**: a callback that turns props and
the pointer into the values the steps read, with `agentFrame` helpers for the arithmetic, and
returns `sys.frame({count})`. `bake` runs once per composition with the compile-time props
(shape, color space) and assembles the steps and their order.

Agents live in canvas units: x from 0 to the aspect ratio, y from 0 to 1, so one unit is the
canvas height and distances are the same in every direction. This module re-exports the whole
agent vocabulary, so one import covers forces, render parts, shape fields and frame helpers.

## Reach for it when

| When | Use |
|---|---|
| a flock, a swarm, drifting motes, iron filings | `agentSim` with an integrator and a `renderAgents` variant |
| agents drawn as arrows or streaks along their heading, colored from calm to excited | `renderAgents.orientedWorld` |
| round motes in one color that twinkle | `renderAgents.pointWorld` |
| a 3D swarm seen in perspective | `renderAgents.volume` |
| particles standing on the layer inside, seen through a camera | `renderAgents.relief` |
| fewer agents on phones without changing the prop's range | the countCap option of `agentSim` |
| a spawn that re-rolls from a seed prop | the initStep returned by bake, with reseed passed to the frame |

## Order

- agentSim
- renderAgents
- renderAgents.orientedWorld
- renderAgents.pointWorld
- renderAgents.volume
- renderAgents.relief

## Example

```ts
import {defineShader, sim, transformColor, transformColorSpace, colorSpaceOptions} from 'shaders/std'
import {tgpu, d, agents} from '@coreroot/gpu/kit'
import {makeCpuValueGetter, readAgentFrame, resolveRenderRes, SIZE_REF_RES} from '@coreroot/gpu/porters'

const {agentSim, renderAgents, force, integrator} = sim.agents

const MAX_AGENTS = 4096
const RES = resolveRenderRes({desktop: 1024, mobile: 640})

// The values every step reads each frame, written from the props in `frame`.
const SimParams = d.struct({
  colA: d.vec4f, colB: d.vec4f,
  count: d.f32, trails: d.f32,
  dt: d.f32, aspect: d.f32, domainX: d.f32,
  maxSpeed: d.f32, maxForce: d.f32,
  perceptionSq: d.f32, sepRadiusSq: d.f32,
  sepW: d.f32, aliW: d.f32, cohW: d.f32,
  cursorX: d.f32, cursorY: d.f32, cursorMode: d.f32, cursorRadius: d.f32, cursorRadiusSq: d.f32, cursorForce: d.f32,
  bodyR: d.f32,
  margin: d.f32, turnForce: d.f32,
  seed: d.f32,
})
// Per-agent state (position + velocity), an excitement value, the two canvases, a trail canvas, the output.
const simLayout = tgpu.bindGroupLayout({
  agents: {storage: d.arrayOf(d.vec4f, MAX_AGENTS), access: 'mutable'},
  agit: {storage: d.arrayOf(d.f32, MAX_AGENTS), access: 'mutable'},
  accumE: {storage: d.arrayOf(d.atomic(d.u32), RES * RES), access: 'mutable'},
  accumS: {storage: d.arrayOf(d.atomic(d.u32), RES * RES), access: 'mutable'},
  trailBuf: {storage: d.arrayOf(d.vec4f, RES * RES), access: 'mutable'},
  params: {uniform: SimParams},
  outTex: {storageTexture: d.textureStorage2d('rgba16float', 'write-only')},
})

// Spawn already in formation, so frame one reads as a murmuration.
const boidsInit = agents.makeClusterFormationInit(simLayout, 'boidsInit')

// The physics: flocking, the cursor, and a soft turn at the edges, folded into one update step.
const boidsUpdate = integrator.steering2d(simLayout, {
  forces: [force.flocking(simLayout, {maxAgents: MAX_AGENTS}), force.cursorSteer(simLayout), force.wallTurn(simLayout)],
  agitation: {rest: 1.5, gain: 0.4545, cool: 1.3},
  cruiseFloor: 0.35,
  wallRestitution: 0.6,
  name: 'boidsUpdate',
})

// The look: comets along the heading, colored from rest to excited, into a trail canvas.
const boidsRender = (shape: string, colorSpace: number) => renderAgents.orientedWorld(simLayout, {
  shape, res: RES, splatRCap: 16, heading: 'velocity', agitation: 'buffer', comet: true,
  ramp: {colorSpace, trails: 'on'},
  names: {splat: 'boidsSplat', resolve: 'boidsResolve'},
})

// Boids: a living murmuration that flashes toward an excited color when the cursor scatters it.
export const Boids = defineShader({
  name: 'Boids',
  usesPointer: true,
  props: {
    colorA: {default: '#8ec5ff', transform: transformColor},
    colorB: {default: '#ff7ad9', transform: transformColor},
    colorSpace: {default: 'oklab', transform: transformColorSpace, compileTime: true, ui: {type: 'select', options: colorSpaceOptions}},
    agentShape: {default: 'arrow', compileTime: true, ui: {type: 'select', options: agents.orientedShapeOptions}},
    count: {default: 2000},
    speed: {default: 2},
    seed: {default: 0},
    size: {default: 1.5},
    trails: {default: 0},
    perception: {default: 0.12},
    cursorStrength: {default: 1.5},
  },
  ...agentSim<typeof SimParams>({
    layout: simLayout,
    params: SimParams,
    maxAgents: MAX_AGENTS,
    countCap: {desktop: MAX_AGENTS, mobile: 1200},
    output: {key: 'outTex', name: 'boidsTexture', size: [RES, RES], format: 'rgba16float'},
    bake: ({getCpuValue}) => {
      const render = boidsRender((getCpuValue('agentShape') as string) || 'arrow', (getCpuValue('colorSpace') as number) ?? 2)
      return {
        pipelines: {
          init: {kernel: boidsInit, threads: 'max'},
          update: {kernel: boidsUpdate, threads: 'agents'},
          splat: render.splat,
          resolve: render.resolve,
        },
        initStep: 'init',
        program: ['update', 'splat', 'resolve'],
      }
    },
    frame: (sys, {getCpuValue}) => {
      const g = makeCpuValueGetter(getCpuValue)
      return (frameParams) => {
        const {dt, aspect, pointerX, pointerY} = readAgentFrame(frameParams)
        const count = sys.resolveCount(g('count', 2000))
        const maxSpeed = 0.13 * g('speed', 2)
        const perception = g('perception', 0.12)
        const colA = getCpuValue('colorA') as {x: number; y: number; z: number; w: number} | undefined
        const colB = getCpuValue('colorB') as {x: number; y: number; z: number; w: number} | undefined
        sys.writeParams({
          colA: d.vec4f(colA?.x ?? 0.56, colA?.y ?? 0.77, colA?.z ?? 1, colA?.w ?? 1),
          colB: d.vec4f(colB?.x ?? 1, colB?.y ?? 0.48, colB?.z ?? 0.85, colB?.w ?? 1),
          count, trails: Math.min(Math.max(g('trails', 0), 0), 1) * 0.92,
          dt, aspect, domainX: Math.max(aspect, 0.01),
          maxSpeed, maxForce: maxSpeed * 3.5,
          perceptionSq: perception * perception,
          sepRadiusSq: (perception * 0.5) * (perception * 0.5),
          sepW: 1.7, aliW: 1.5, cohW: 1,
          cursorX: pointerX * aspect, cursorY: pointerY, cursorMode: 2,
          cursorRadius: 0.22, cursorRadiusSq: 0.22 * 0.22,
          cursorForce: g('cursorStrength', 1.5) * maxSpeed * 12,
          bodyR: Math.min(Math.max(g('size', 1.5), 0.5), 3) * 2 / SIZE_REF_RES,
          margin: 0.07, turnForce: maxSpeed * 2.2,
          seed: g('seed', 0),
        })
        return sys.frame({count, reseed: g('seed', 0)})
      }
    },
    fragment: {output: 'boidsTexture', fallback: 'transparent'},
  }),
})
```
