# Light

Words for drawing light: glows, rays, beams, flares and halos, plus the frames they are drawn
in. A **frame** is the coordinate system a light evaluates in. `radialFrame` gives each pixel
its distance and angle from a center; `beamFrame` gives its distance along and across a beam.
**Parts** such as `glowPoint`, `rayLobes` and `featherMask` shape a light over a frame and
return a *field*, a 0–1 number per pixel. `withFrame` evaluates a part in a frame,
`modulate` multiplies parts (a mask over a pattern), and `additive` sums finished lights.

A light field becomes color at the end of the recipe. `coverageMix` and `coverageOver` color
a field over a background as a finished `paint:`. `heatRamp` turns a heat field into
overexposed color, and `screenGlow` shines it onto the layer inside as an `effect:`. Emitted
light drawn over transparency closes with `emissiveAlpha` from `compose`.

## Reach for it when

| When | Use |
|---|---|
| a soft round glow at a position | `glowAt` |
| a glow, rays or a halo around a center prop | `radialFrame` + `withFrame` with `glowPoint`, `rayLobes`, `ringBand` |
| a light that fades at a radius | `featherMask` |
| light spilling from an emitter with a real falloff | `emitterFalloff` |
| volumetric rays or god rays | `noiseRays` in a `seamlessRadialFrame` |
| coloring a light field over a background | `coverageMix` or `coverageOver` |
| a beam anchored at a point aimed at the canvas | `beamFrame` with `beamBloom`, `beamStreaks` |
| a beam between two points, tapered end to end | `segmentProject` + `lightBeam` |
| a camera lens flare | `flareFrame` + the `flare*` parts, closed with `flareComposite` |
| a shiny surface from a normal field | `shine` |
| turning a heat field into overexposed color | `heatRamp` |
| darkening the corners | `vignetteMask` |
| a glow that behaves like exposure over a child | `screenGlow` |

## Order

- glowAt
- glowSpot
- emitterFalloff
- radialFrame
- seamlessRadialFrame
- beamFrame
- withFrame
- modulate
- glowPoint
- featherMask
- rayLobes
- softRayLobes
- raySpikes
- ringBand
- noiseRays
- rayBands
- shine
- domeNormal
- additive
- heatRamp
- vignetteMask
- screenGlow
- coverageMix
- coverageOver
- slowDrift
- beamBloom
- beamStreaks
- streakBand
- lightBeam
- segmentProject
- flareFrame
- lensGhost
- flareHalo
- flareStarburst
- flareStreak
- flareGlare
- flareCore
- flareComposite
- dithered

## Example

```ts
import {defineShader, p, light, compose, math, transformColor, transformPosition} from 'shaders/std'

const {radialFrame, withFrame, modulate, glowPoint, softRayLobes, featherMask} = light

// A sun: soft rays spinning around a draggable center, feathered at a radius, drawn as emitted light.
export const Sun = defineShader({
  name: 'Sun',
  animatedTime: {speed: 'speed'},
  props: {
    center: {default: {x: 0.5, y: 0.5}, transform: transformPosition},
    color: {default: '#ffd27a', transform: transformColor},
    radius: {default: 0.5},
    rays: {default: 12},
    softness: {default: 0.6},
    speed: {default: 1},
  },
  paint: (params) => {
    const coverage = withFrame(
      radialFrame(p('center')),
      modulate(softRayLobes({count: p('rays'), softness: p('softness'), spin: 0.2}), glowPoint({sharpness: 6}), featherMask({radius: p('radius'), feather: 0.4})),
      'sun',
    )(params)
    const rgb = math.mul(params.uniforms.color.member('rgb'), coverage)
    return compose.emissiveAlpha(rgb)
  },
})
```
