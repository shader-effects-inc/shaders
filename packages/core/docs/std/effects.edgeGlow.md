# Edge glow

A band of light hugging the edge of any shape, with coloured spots racing around it. The words
live inside a `shapedSurface` surface function (from `materials`), which hands them the
shape's **distance field**: for every pixel, how far it is from the shape's edge, negative
inside. `edgeGlowBand` turns that distance into the glowing band (with drifting smoke) and
also gives the angle around the shape. `orbitSpotsAccum` places one colour's spots along that
angle. The spots of every colour are folded together, starting from `edgeGlowAccumZero` and
adding one colour at a time with the step from `edgeGlowAccumColorFor`, and `edgeGlowCompose`
turns the fold into the final colour with bloom. `heartbeatPulse` is the beat the band and the
spots share.

These are GPU functions: invoke one with `call(fn, 'name', [args])`. The result is straight
alpha over a transparent background, so it composites over whatever sits behind the shape.

## Reach for it when

| When | Use |
|---|---|
| a glowing outline around a 2D, SVG or 3D shape | `edgeGlowBand` |
| coloured lights chasing each other around that outline | `orbitSpotsAccum` |
| several colours of light that overlap without going muddy | `edgeGlowAccumColorFor` with `edgeGlowAccumZero` |
| the colours from a `stops` prop, one spot orbit each | `colorAtIndex` |
| a pulse the band and the spots beat to | `heartbeatPulse` |
| the final colour, with bloom | `edgeGlowCompose` |

## Order

- edgeGlowBand
- orbitSpotsAccum
- heartbeatPulse
- edgeGlowAccumZero
- edgeGlowAccumColorFor
- edgeGlowCompose
- colorAtIndex

## Example

```ts
import {defineShader, effects, materials, math, call, animatedTime, transformPosition, transformColor, transformColorSpace} from 'shaders/std'
import type {Expr} from 'shaders/std'

const {heartbeatPulse, edgeGlowBand, orbitSpotsAccum, edgeGlowAccumZero, edgeGlowAccumColorFor, edgeGlowCompose} = effects.edgeGlow

// Halo: two colours of light chasing each other around the edge of a shape, with smoke and bloom.
export const Halo = defineShader({
  name: 'Halo',
  animatedTime: {speed: 'speed'},
  props: {
    center: {default: {x: 0.5, y: 0.5}, transform: transformPosition},
    scale: {default: 1},
    rotation: {default: 0},
    shape: {default: JSON.stringify({type: 'circleSDF', radius: 0.35}), ui: {type: 'shape', label: 'Shape', group: 'Shape'}},
    shapeSdfUrl: {default: '', compileTime: true},
    shapeType: {default: '', compileTime: true},
    colorA: {default: '#7b2ff7', transform: transformColor},
    colorB: {default: '#00e0ff', transform: transformColor},
    colorSpace: {default: 'oklch', transform: transformColorSpace, compileTime: true},
    thickness: {default: 0.25},
    softness: {default: 0.5},
    intensity: {default: 0.6},
    bloom: {default: 0.3},
    spots: {default: 3},
    spotSize: {default: 0.5},
    pulse: {default: 0},
    smoke: {default: 0.3},
    smokeSize: {default: 0.5},
    speed: {default: 1},
    seed: {default: 0},
  },
  ...materials.shapedSurface({
    chord: 'firstLobe',
    stencil: 'centre',
    surface: (frame, params) => {
      const {uniforms} = params
      const t = animatedTime(params)
      // One beat per pixel, shared by the band's smoke and every spot orbit.
      const beat = math.local(call(heartbeatPulse, 'heartbeatPulse', [t]), 'beat')
      const band = math.local(call(edgeGlowBand, 'edgeGlowBand', [
        frame.surf0!.member('r'), frame.sdfUV!, params.ctx.viewportSize, uniforms.scale, uniforms.thickness, uniforms.softness,
        uniforms.smokeSize, uniforms.smoke, t, uniforms.pulse, beat,
      ]), 'band')
      const spotsFor = (colorIndex: number): Expr => call(orbitSpotsAccum, 'orbitSpotsAccum', [
        band.member('y'), band.member('x'), t, beat, uniforms.spots, uniforms.spotSize, uniforms.pulse,
        uniforms.intensity, uniforms.softness, uniforms.seed, math.float(colorIndex),
      ])
      // Fold the two colours together in the chosen colour space, then compose with bloom.
      const accumColor = edgeGlowAccumColorFor((params.propValues.colorSpace as number) ?? 0)
      let state = call(edgeGlowAccumZero, 'edgeGlowAccumZero', [])
      state = call(accumColor, 'edgeGlowAccumColor', [state, uniforms.colorA, spotsFor(0)])
      state = call(accumColor, 'edgeGlowAccumColor', [state, uniforms.colorB, spotsFor(1)])
      return call(edgeGlowCompose, 'edgeGlowCompose', [state, uniforms.bloom])
    },
  }),
})
```
