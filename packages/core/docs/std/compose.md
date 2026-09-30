# Layering & output

The words that finish a `paint:` function. `layers` stacks several colors into one, back to
front, each with an optional opacity and an optional factor it is seen through. `layered`
builds N variations of one recipe from a data list and sums them, which is how curtains,
cloud banks and echo trails get their depth cheaply.

Light is emitted, so it should composite *over* what is behind the layer rather than replace
it. `emissiveAlpha` turns an rgb light value into a color with alpha that does exactly that.
`dithered` adds a trace of noise to the final color so long soft ramps show no banding. Use
them last: stack with `layers`, close with `emissiveAlpha`, then `dithered`.

## Reach for it when

| When | Use |
|---|---|
| several glows or paints stacked into one image | `layers` |
| the same recipe repeated with different offsets and weights | `layered` |
| a glow that should composite over transparency | `emissiveAlpha` |
| visible steps in a long soft gradient | `dithered` |

## Order

- layers
- emissiveAlpha
- dithered
- layered

## Example

```ts
import {defineShader, p, light, compose, math, transformColor, transformPosition} from 'shaders/std'

const {radialFrame, withFrame, glowPoint} = light
const {layers, emissiveAlpha, dithered} = compose

// Two soft glows, the second screened over the first, drawn as emitted light and dithered.
export const TwinGlow = defineShader({
  name: 'TwinGlow',
  animatedTime: {speed: 'speed'},
  props: {
    centerA: {default: {x: 0.35, y: 0.5}, transform: transformPosition},
    centerB: {default: {x: 0.65, y: 0.5}, transform: transformPosition},
    colorA: {default: '#ff7a18', transform: transformColor},
    colorB: {default: '#3b82f6', transform: transformColor},
    sharpness: {default: 6},
    speed: {default: 0},
  },
  paint: (params) => {
    const glowA = withFrame(radialFrame(p('centerA')), glowPoint({sharpness: p('sharpness')}), 'glowA')(params)
    const glowB = withFrame(radialFrame(p('centerB')), glowPoint({sharpness: p('sharpness')}), 'glowB')(params)
    const rgb = layers([
      {paint: math.mul(params.uniforms.colorA.member('rgb'), glowA)},
      {paint: math.mul(params.uniforms.colorB.member('rgb'), glowB), blend: 'screen'},
    ])
    return dithered(emissiveAlpha(rgb), params)
  },
})
```
