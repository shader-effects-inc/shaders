# Volumes

Words for filling a shape with something you look through: smoke, gas, tinted glass, a
starfield behind a window. They work inside a shape-effect material, the `surface:` of
`shapedSurface` from `materials`. An **interior ray** is the line of sight through the shape:
where it enters, how far it travels inside, and where it is at any depth. A **medium** is a
density at each point inside. A **march** walks the ray front to back, adding up what each
sample glows and how much it dims what lies behind it.

Start with `surfaceField` and `viewRay`, build the ray with `interiorRay`, then call
`volumeMarch` and sample `turbulentMedium` at `ray.at(z)` inside its callback. `parallaxPlane`
places a backdrop at a depth behind the gas. The march returns a color and a transmittance.
Layer the color over anything behind it multiplied by the transmittance, take the alpha
from `silhouette`, and wrap it all in `guarded` with `insideShape` so pixels outside the
shape stay cheap.

## Reach for it when

| When | Use |
|---|---|
| smoke or gas sealed inside a shape | `interiorRay` and `volumeMarch` over `turbulentMedium` |
| a backdrop or starfield seen through the shape | `parallaxPlane` behind the march |
| a medium that tints what lies behind it | a per-channel absorb in `volumeMarch` |
| cloud density with free dust and wisp detail | `turbulentMedium` |

## Order

- interiorRay
- volumeMarch
- turbulentMedium
- parallaxPlane

## Example

```ts
import {defineShader, animatedTime, materials, volume, math, transformColor, transformPosition} from 'shaders/std'

const {shapedSurface, surfaceField, viewRay, hashNoise, silhouette, guarded, insideShape} = materials
const {interiorRay, turbulentMedium, volumeMarch} = volume
const {float, local, mul, smoothstep, splat3, sub, vec4} = math

const STEPS = 6

// Smoke sealed inside any shape: a six-step march through billowing density, glowing in the smoke color.
export const Smoke = defineShader({
  name: 'Smoke',
  role: 'shapeEffect',
  species: 'custom',
  animatedTime: {speed: 'speed'},
  props: {
    center: {default: {x: 0.5, y: 0.5}, transform: transformPosition},
    scale: {default: 1},
    rotation: {default: 0},
    color: {default: '#ffb27a', transform: transformColor},
    density: {default: 1},
    gasScale: {default: 1},
    edgeSoftness: {default: 0.01},
    speed: {default: 1},
    shape: {default: JSON.stringify({type: 'sphere3D', radius: 0.35}), ui: {type: 'shape', label: 'Shape'}},
    shapeSdfUrl: {default: '', compileTime: true},
    shapeType: {default: '', compileTime: true},
  },
  ...shapedSurface({
    stencil: 'centre',
    centreTap: 'fast',
    surface: (frame, params) => {
      const u = params.uniforms
      const field = surfaceField(frame, params, {scale: u.scale})
      const view = viewRay(params, field)
      // The shape's own coordinates, centred, as the point the ray enters at.
      const origin = {x: sub(frame.sdfUV!.member('x'), 0.5), y: sub(frame.sdfUV!.member('y'), 0.5)}
      const ray = interiorRay(field, frame, {origin, view, shear: 0.5})
      const drift = local(mul(animatedTime(params), 0.15), 'drift')
      const jitter = mul(hashNoise(mul(origin.x, 311.7), mul(origin.y, 173.3)), 0.15 / STEPS)
      const march = volumeMarch(
        {steps: STEPS, entry: ray.entry, length: ray.length, jitter, emissionGain: float(4 / STEPS)},
        ({i, z}) => {
          const at = ray.at(z)
          const gas = turbulentMedium({x: at.x, y: at.y, z}, {frequency: mul(u.gasScale, 3), drift, billow: float(0.4), hint: `${i}`})
          const dens = local(smoothstep(0, 0.6, gas.density), `dens${i}`)
          return {emit: mul(u.color.member('rgb'), dens), absorb: splat3(mul(dens, mul(u.density, 3 / STEPS)))}
        },
      )
      const smoke = vec4(march.color, silhouette(field, u.edgeSoftness))
      return guarded(insideShape(field.sdf, field.pxH), smoke, vec4(0, 0, 0, 0), 'smoke')
    },
  }),
})
```
