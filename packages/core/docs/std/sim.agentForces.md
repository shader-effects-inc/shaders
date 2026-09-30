# Agent forces

What pushes, turns and carries each agent. The physics of an agent simulation is a list of
parts folded into one update step by an **integrator**. A `force` returns a push for one
agent; a `torque` returns a turn; a `field` samples something at the agent's position (the
magnet at the cursor, a home on a grid); a `drift` is a built-in velocity. Every part is built
over your layout, which names the per-agent buffers and the per-frame values it reads, so the
same part works in any layer that provides those names.

Six kinds of agent, six integrators. `integrator.forces3d` moves a 3D swarm held in a shape
(`force.pressure`, `force.containment`, `force.cursorXY`, `force.gravity`, `force.turbulence`).
`integrator.steering2d` moves a flock (`force.flocking`, `force.cursorSteer`, `force.wallTurn`).
`integrator.orientation2d` turns agents in place (`torque.alignToField`, `torque.rest`, over
`field.dipoleOrRadial` and `field.jitteredGridHome`). `integrator.drift2d` glides motes
(`drift.variedHeading`, `force.cursorGust`). `integrator.relief` stands particles on the
child's image (`channel`, `force.cursorInView`). `integrator.advect2d` carries particles along
a fluid. Pick the integrator, list the parts, and register the result as the update step in
`agentSim`'s `bake`.

Drag arrives as a per-frame multiplier the frame writes each frame: compute it with
`agentFrame.expDecay` so it does not depend on frame rate.

## Reach for it when

| When | Use |
|---|---|
| a flock that separates, aligns and coheres | `integrator.steering2d` with `force.flocking` |
| the cursor attracts or scatters a flock | `force.cursorSteer` |
| a swarm that fills a shape evenly and stays inside it | `integrator.forces3d` with `force.pressure` and `force.containment` |
| iron filings that swing onto a magnetic field | `integrator.orientation2d` with `torque.alignToField`, `torque.rest`, `field.dipoleOrRadial` |
| motes that glide in one direction with variety | `integrator.drift2d` with `drift.variedHeading` |
| a puff from the cursor on drifting motes | `force.cursorGust` |
| particles that rise by the brightness of the layer inside | `integrator.relief` with `channel` |
| particles carried along a fluid's velocity | `integrator.advect2d` |

## Order

- integrator
- integrator.steering2d
- integrator.forces3d
- integrator.orientation2d
- integrator.drift2d
- integrator.relief
- integrator.advect2d
- force
- force.flocking
- force.cursorSteer
- force.wallTurn
- force.pressure
- force.containment
- force.cursorXY
- force.gravity
- force.turbulence
- force.cursorGust
- force.cursorInView
- torque
- torque.alignToField
- torque.rest
- field
- field.dipoleOrRadial
- field.jitteredGridHome
- drift
- drift.variedHeading
- channel
- FieldSample

## Example

```ts
import {defineShader, sim, transformColor} from 'shaders/std'
import {tgpu, d, agents} from '@coreroot/gpu/kit'
import {makeCpuValueGetter, readAgentFrame, resolveRenderRes, SIZE_REF_RES} from '@coreroot/gpu/porters'

const {agentSim, renderAgents, torque, field, integrator, agentFrame} = sim.agents

const MAX_FILINGS = 12288
const RES = resolveRenderRes({desktop: 1024, mobile: 768})
const OVERSCAN = 0.14

// The values the torques, the field and the home read each frame.
const SimParams = d.struct({
  colA: d.vec4f, colB: d.vec4f,
  count: d.f32,
  dt: d.f32, aspect: d.f32, domainX: d.f32,
  cursorX: d.f32, cursorY: d.f32, axisX: d.f32, axisY: d.f32,
  fieldType: d.f32, restMode: d.f32,
  strength: d.f32, reachSq: d.f32,
  alignK: d.f32, damping: d.f32, restK: d.f32, pullK: d.f32, homeK: d.f32,
  omegaRef: d.f32, agitCool: d.f32,
  gridCols: d.f32, cellW: d.f32, cellH: d.f32, jitter: d.f32,
  bodyR: d.f32,
})
// Per-agent state (offset from home, angle, spin), excitement, the two canvases, the output.
const simLayout = tgpu.bindGroupLayout({
  agents: {storage: d.arrayOf(d.vec4f, MAX_FILINGS), access: 'mutable'},
  agit: {storage: d.arrayOf(d.f32, MAX_FILINGS), access: 'mutable'},
  accumE: {storage: d.arrayOf(d.atomic(d.u32), RES * RES), access: 'mutable'},
  accumS: {storage: d.arrayOf(d.atomic(d.u32), RES * RES), access: 'mutable'},
  params: {uniform: SimParams},
  outTex: {storageTexture: d.textureStorage2d('rgba16float', 'write-only')},
})

const filingsInit = agents.makeRestingDirectorInit(simLayout, 'filingsInit')

// The physics: every filing rests on a jittered grid, samples the magnet at the cursor, and
// swings onto its field line, with a weak pull back to its rest angle where the field is faint.
const filingsHome = field.jitteredGridHome(simLayout, {overscan: OVERSCAN})
const filingsUpdate = integrator.orientation2d(simLayout, {
  home: filingsHome,
  fieldAt: field.dipoleOrRadial(simLayout),
  torques: [torque.alignToField(simLayout), torque.rest(simLayout)],
  maxOffset: 0.12,
  name: 'filingsUpdate',
})

// The look: slivers drawn along each filing's own angle, colored calm to excited.
const filingsRender = renderAgents.orientedWorld(simLayout, {
  shape: 'streak', res: RES, splatRCap: 14, heading: 'angle', agitation: 'buffer', comet: false, home: filingsHome,
  ramp: {colorSpace: 2, trails: 'none'},
  names: {splat: 'filingsSplat', resolve: 'filingsResolve'},
})

// MagneticFilings: iron filings that reveal the field lines of a magnet at the cursor.
export const MagneticFilings = defineShader({
  name: 'MagneticFilings',
  usesPointer: true,
  props: {
    colorA: {default: '#85929e', transform: transformColor},
    colorB: {default: '#ff9d5c', transform: transformColor},
    count: {default: 5000},
    strength: {default: 1},
    reach: {default: 0.35},
    response: {default: 0.5},
    size: {default: 1},
  },
  ...agentSim<typeof SimParams>({
    layout: simLayout,
    params: SimParams,
    maxAgents: MAX_FILINGS,
    output: {key: 'outTex', name: 'filingsTexture', size: [RES, RES], format: 'rgba16float'},
    bake: () => ({
      pipelines: {
        init: {kernel: filingsInit, threads: 'max'},
        update: {kernel: filingsUpdate, threads: 'agents'},
        splat: filingsRender.splat,
        resolve: filingsRender.resolve,
      },
      initStep: 'init',
      program: ['update', 'splat', 'resolve'],
    }),
    frame: (sys, {getCpuValue}) => {
      const g = makeCpuValueGetter(getCpuValue)
      // The magnet's axis follows where the cursor has been travelling.
      const motionAxis = agentFrame.createMotionAxis({smoothing: 0.1, teleport: 0.25})
      return (frameParams) => {
        const {dt, aspect, pointerX, pointerY} = readAgentFrame(frameParams)
        const domainX = Math.max(aspect, 0.01)
        const count = sys.resolveCount(g('count', 5000))
        const {cols, cellW, cellH} = agentFrame.fitJitteredGrid(count, domainX, OVERSCAN)
        const cursorX = pointerX * aspect
        const cursorY = pointerY
        const axis = motionAxis.update(cursorX, cursorY)
        const reach = g('reach', 0.35)
        const colA = getCpuValue('colorA') as {x: number; y: number; z: number; w: number} | undefined
        const colB = getCpuValue('colorB') as {x: number; y: number; z: number; w: number} | undefined
        sys.writeParams({
          colA: d.vec4f(colA?.x ?? 0.52, colA?.y ?? 0.57, colA?.z ?? 0.64, colA?.w ?? 1),
          colB: d.vec4f(colB?.x ?? 1, colB?.y ?? 0.62, colB?.z ?? 0.36, colB?.w ?? 1),
          count,
          dt, aspect, domainX,
          cursorX, cursorY, axisX: axis.x, axisY: axis.y,
          fieldType: 0, restMode: 0,
          strength: g('strength', 1), reachSq: reach * reach,
          alignK: 45, damping: 3 + Math.min(Math.max(g('response', 0.5), 0), 1) * 9,
          restK: 6, pullK: 0.25, homeK: 2.5,
          omegaRef: 6, agitCool: 1,
          gridCols: cols, cellW, cellH, jitter: 0.85,
          bodyR: Math.min(Math.max(g('size', 1), 0.5), 3) * 1.7 / SIZE_REF_RES,
        })
        return sys.frame({count})
      }
    },
    fragment: {output: 'filingsTexture', fallback: 'transparent'},
  }),
})
```
