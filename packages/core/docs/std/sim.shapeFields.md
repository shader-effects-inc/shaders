# Shape fields

The shape a swarm is held inside. `force.containment` needs one thing from a shape: the signed
distance to its surface at any 3D point, negative inside and positive outside. `shapeField`
builds that distance function from whatever shape the layer was given, so one integrator serves
every shape kind: `analyticExtruded` gives a flat shape (circle, polygon, star) a depth;
`svgExtruded` does the same for an SVG outline uploaded as a distance texture; `analytic3d`
wraps a true 3D solid; `svgLifted3d` wraps an SVG lifted into 3D. The last two rotate with the
shape's live rotation, so the swarm follows the surface as it turns.

Coordinates are shape-local: the shape sits in a cube from −0.5 to 0.5 with y up, and the
layer's center, scale and rotation are applied when the swarm is drawn. Pick the builder in
`agentSim`'s `bake` from the shape props, hand one field to `force.containment`, and the
integrator never needs to know which kind it was. The flat builders read the shape's sub-props
and half-depth from the values struct each frame.

## Reach for it when

| When | Use |
|---|---|
| a swarm held inside a flat shape with some depth | `shapeField.analyticExtruded` |
| a swarm held inside an SVG outline with some depth | `shapeField.svgExtruded` |
| a swarm inside a sphere, torus or other 3D solid that rotates | `shapeField.analytic3d` |
| a swarm inside an SVG lifted into 3D | `shapeField.svgLifted3d` |
| the force that reads the field | `force.containment` |

## Order

- shapeField
- shapeField.analyticExtruded
- shapeField.svgExtruded
- shapeField.analytic3d
- shapeField.svgLifted3d

## Example

```ts
import {defineShader, sim, transformColor} from 'shaders/std'
import {tgpu, d, sdf, agents} from '@coreroot/gpu/kit'
import {makeCpuValueGetter, readAgentFrame, resolveRenderRes, SIZE_REF_RES} from '@coreroot/gpu/porters'

const {agentSim, renderAgents, force, integrator, agentFrame, shapeField} = sim.agents

const MAX_PARTICLES = 16000
const DGRID = 32
const DOMAIN = 1.0
const OUT_RES = resolveRenderRes({desktop: 1024, mobile: 768})
const MAX_SPEED = 3

const SimParams = d.struct({
  colA: d.vec4f, colB: d.vec4f,
  dt: d.f32, time: d.f32, spread: d.f32, agitation: d.f32,
  dragMul: d.f32, gravX: d.f32, gravY: d.f32, halfDepth: d.f32,
  cursorX: d.f32, cursorY: d.f32, cursorForce: d.f32, cursorRadSq: d.f32,
  saRadius: d.f32, saSides: d.f32, saRounding: d.f32, saInnerRatio: d.f32,
  saRotation: d.f32, saHeight: d.f32, saOffset: d.f32, saAperture: d.f32,
  centerX: d.f32, centerYv: d.f32, scale: d.f32, rotC: d.f32, rotS: d.f32, aspect: d.f32,
  size: d.f32, exposure: d.f32, softness: d.f32, speedColorK: d.f32,
  omegaX: d.f32, omegaY: d.f32, omegaZ: d.f32, entrain: d.f32,
  gridOffX: d.f32, gridOffY: d.f32, gridOffZ: d.f32,
})
const simLayout = tgpu.bindGroupLayout({
  pos: {storage: d.arrayOf(d.vec4f, MAX_PARTICLES), access: 'mutable'},
  vel: {storage: d.arrayOf(d.vec4f, MAX_PARTICLES), access: 'mutable'},
  dens: {storage: d.arrayOf(d.atomic(d.u32), DGRID * DGRID * DGRID), access: 'mutable'},
  accumE: {storage: d.arrayOf(d.atomic(d.u32), OUT_RES * OUT_RES), access: 'mutable'},
  accumS: {storage: d.arrayOf(d.atomic(d.u32), OUT_RES * OUT_RES), access: 'mutable'},
  params: {uniform: SimParams},
  outTex: {storageTexture: d.textureStorage2d('rgba16float', 'write-only')},
})

const swarmInit = agents.makeBallCloudInit(simLayout, 'swarmInit')
// Even filling: particles are counted into a density grid and slide away from crowding.
const pressure = force.pressure(simLayout, {gridDim: DGRID, domain: DOMAIN, names: {clear: 'swarmClearDensity', splat: 'swarmDensity'}})

// A hexagon with depth, as the field the containment force reads. Its radius, sides and rounding
// arrive through the values struct each frame.
const hexField = shapeField.analyticExtruded(simLayout, sdf.buildAnalyticSdfFn('polygonSDF'))
const swarmUpdate = integrator.forces3d(simLayout, {
  forces: [
    pressure.force,
    force.containment(simLayout, {
      field: hexField,
      gradEps: 0.02,
      wall: {featherFrom: -0.10, featherTo: 0.01, base: 1.1, springK: 22},
      recall: {from: DOMAIN * 0.85, to: DOMAIN * 1.1, k: 9},
      homing: {from: 0.12, to: 0.45, k: 5},
      entrainment: {featherHalf: 0.05},
    }),
    force.cursorXY(simLayout),
    force.gravity(simLayout),
    force.turbulence(simLayout, {posScale: 193, timeX: 31.7, timeY: 27.3, seedScale: 997, gain: 5}),
  ],
  maxSpeed: MAX_SPEED,
  posClamp: 1.9,
  name: 'swarmUpdate',
})
const swarmRender = renderAgents.volume(simLayout, {
  shape: 'dot', outRes: OUT_RES, maxSplatSize: 6, extQ: 2.6,
  ramp: {colorSpace: 0, exposure: true, trails: 'none'},
  names: {splat: 'swarmSplat', glow: 'swarmGlowProfile', profile: 'swarmProfile', resolve: 'swarmResolve'},
})

// HexSwarm: particles that fill a hexagon evenly and scatter when the cursor pushes through them.
export const HexSwarm = defineShader({
  name: 'HexSwarm',
  usesPointer: true,
  props: {
    colorA: {default: '#ffffff', transform: transformColor},
    colorB: {default: '#ff7ad9', transform: transformColor},
    count: {default: 4000},
    radius: {default: 0.35},
    depth: {default: 0.18},
    spread: {default: 1},
    damping: {default: 0.4},
    mouseInfluence: {default: 1.2},
  },
  ...agentSim<typeof SimParams>({
    layout: simLayout,
    params: SimParams,
    maxAgents: MAX_PARTICLES,
    countCap: {desktop: MAX_PARTICLES, mobile: 6000},
    output: {key: 'outTex', name: 'particleTexture', size: [OUT_RES, OUT_RES], format: 'rgba16float'},
    bake: () => ({
      pipelines: {
        init: {kernel: swarmInit, threads: 'max'},
        clearDensity: {kernel: pressure.clearKernel, threads: 'fixed', size: [pressure.cells]},
        density: {kernel: pressure.splatKernel, threads: 'agents'},
        update: {kernel: swarmUpdate, threads: 'agents'},
        splat: swarmRender.splat,
        resolve: swarmRender.resolve,
      },
      initStep: 'init',
      program: ['clearDensity', 'density', 'update', 'splat', 'resolve'],
    }),
    frame: (sys, {getCpuValue}) => {
      const g = makeCpuValueGetter(getCpuValue)
      let time = 0
      let frameIdx = 0
      const CELL = (2 * DOMAIN) / DGRID
      return (frameParams) => {
        const {dt, aspect, pointerX, pointerY} = readAgentFrame(frameParams, 1)
        time += dt
        const centerX = 0.5, centerYv = 0.5, scale = 1, rotC = 1, rotS = 0
        // The pointer in the shape's own coordinates, so the cursor force acts where the swarm lives.
        const cursor = agentFrame.pointerToShapeLocal({pointerX, pointerY, centerX, centerYv, scale, rotC, rotS, aspect})
        const gridOff = agentFrame.r3SubCellOffset(frameIdx++, CELL)
        const colA = getCpuValue('colorA') as {x: number; y: number; z: number; w: number} | undefined
        const colB = getCpuValue('colorB') as {x: number; y: number; z: number; w: number} | undefined
        sys.writeParams({
          colA: d.vec4f(colA?.x ?? 1, colA?.y ?? 1, colA?.z ?? 1, colA?.w ?? 1),
          colB: d.vec4f(colB?.x ?? 1, colB?.y ?? 0.48, colB?.z ?? 0.85, colB?.w ?? 1),
          dt, time,
          spread: g('spread', 1),
          agitation: 0.12,
          dragMul: agentFrame.expDecay(2.2 + g('damping', 0.4) * 6, dt),
          gravX: 0, gravY: 0,
          halfDepth: Math.max(g('depth', 0.18), 0.01) * 0.5,
          cursorX: cursor.x, cursorY: cursor.y,
          cursorForce: g('mouseInfluence', 1.2) * 3,
          cursorRadSq: 0.22 * 0.22,
          saRadius: g('radius', 0.35), saSides: 6, saRounding: 0, saInnerRatio: 0.4,
          saRotation: 0, saHeight: 0.25, saOffset: 0.2, saAperture: 270,
          centerX, centerYv, scale, rotC, rotS, aspect,
          size: 2.2 * OUT_RES / SIZE_REF_RES,
          exposure: 1,
          softness: 0.5,
          speedColorK: 2 / MAX_SPEED,
          omegaX: 0, omegaY: 0, omegaZ: 0, entrain: 0,
          gridOffX: gridOff.x, gridOffY: gridOff.y, gridOffZ: gridOff.z,
        })
        return sys.frame({count: sys.resolveCount(g('count', 4000))})
      }
    },
    fragment: {output: 'particleTexture', fallback: 'transparent'},
  }),
})
```
