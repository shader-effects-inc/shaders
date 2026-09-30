# Materials & surfaces

Words that shade a surface so a shape reads as glass, metal, plastic or stone. A material is
three things: a **normal** (which way the surface faces at this pixel), a **light** (where it
comes from) and a **response** (how the surface answers: diffuse, glint, mirror, tint).
`shapedSurface` turns a shape prop into a surface you can shade. It hands your `surface`
function a frame with the shape's distance field, and `surfaceField` binds what you read from
it: the signed distance, the pixel size, the neighbour taps.

From there a material is a recipe read top to bottom. A normal from `geometricNormal`, bent
by `nudgeNormal` or `tiltNormal` for relief. A light from `keyLightAt` (an angle) or
`pointLightFrom` (a position). A response from `lambert`, `sharpGlint`, `dualLobeGlint`,
`schlickFresnel`, `beerLambert`. A finish from `neutralTone` and `silhouette`, closed with
`guarded` around `insideShape` so pixels outside the shape pay nothing. The same words light
anything with a slope, not only shapes: `fdSlope` of a noise field feeds `nudgeNormal`, and
`keyLightAt` with `lambert` shades it.

Two conventions to hold. The viewer looks along +z, so a normal that faces the viewer has
negative z and a key light on the viewer's side has a negative elevation (−0.6 raking to
−0.9 frontal). And a trailing `hint` argument only names a value in the compiled shader for
debugging. Leave it out.

## Reach for it when

| When | Use |
|---|---|
| a shape prop should become a shaded surface | `shapedSurface`, then `surfaceField` as the first line |
| the surface normal of whatever shape is active | `geometricNormal` |
| relief from a height field or noise | `fdSlope` or `perlinSlope` into `nudgeNormal` |
| brushed or woven relief along a direction | `tiltNormal` with `grainNoise` |
| a light from an angle prop | `keyLightAt` |
| a light from a position prop | `placementPoint` into `pointLightFrom`, faded by `inverseSquare` |
| soft diffuse shading | `lambert` |
| a highlight that follows a sharpness slider | `sharpGlint` |
| a highlight with a hot core and a wide halo | `dualLobeGlint` |
| a highlight stretched along a grain | `anisoSpecular` with `wardAlphas` |
| brighter or more mirror-like rims | `fresnelBoost`, `schlickFresnel` over `grazingOf` or `grazingFlat` |
| a studio reflection in a metal | `viewRay`, `reflect`, `studioSoftboxes`, smeared by `smearAlong` |
| glass, ice or water that darkens with thickness | `opticalThickness` into `beerLambert` |
| a rainbow sheen | `cosineRainbow` |
| stars, glitter or dust | `pointStars` with `twinkle` |
| a soft edge and nothing drawn outside the shape | `silhouette` inside `guarded` on `insideShape` |
| bright light brought back into range | `neutralTone` for a surface, `exposureTone` for emitted light |
| film grain or per-pixel jitter | `sensorGrain`, `hashNoise`, `interleavedNoise` |
| a rim glow or contact shadow outside the shape | `nearestEdge`, `continuedField` |

## Order

- shapedSurface
- surfaceField
- geometricNormal
- marchedNormal
- nudgeNormal
- tiltNormal
- tiltAlong
- fdSlope
- perlinSlope
- fieldSlope
- surfacePattern
- keyLightAt
- keyLightXY
- lightVec3
- placementPoint
- pointLightFrom
- inverseSquare
- lambert
- sharpGlint
- dualLobeGlint
- anisoSpecular
- wardAlphas
- viewRay
- grazingOf
- grazingFlat
- fresnelBoost
- schlickFresnel
- studioSoftboxes
- smearAlong
- opticalThickness
- beerLambert
- cosineRainbow
- tintRamp
- neutralTone
- exposureTone
- silhouette
- insideShape
- guarded
- surfaceNoise
- surfaceNoiseAt
- volumeNoiseAt
- cellNoiseAt
- valueNoise
- hashNoise
- interleavedNoise
- grainNoise
- flowWarp
- sensorGrain
- pointStars
- twinkle
- shellRefract
- rotationSensor
- nearestEdge
- continuedField

## Example

```ts
import {defineShader, p, uniformOf, paint, frames, materials, compose, math, transformColor, transformColorSpace, colorSpaceOptions} from 'shaders/std'

const {add, mul, dot, vec3, vec4} = math
const {noiseField, warped, scaledVolume, tone, pair} = paint
const {surfaceOf, direction} = frames
const {fdSlope, nudgeNormal, keyLightAt, lambert, dualLobeGlint, neutralTone} = materials

// One height field drives everything: warped noise on a zoomable slab.
const height = scaledVolume(
  warped(noiseField('mx3'), {amount: p('warp'), amountScale: 4, timeScale: 0.12}),
  {scale: p('scale')},
)

// Silk: the field's slope becomes a normal, a key light shades it, a two-lobe sheen catches the folds.
export const SilkTide = defineShader({
  name: 'SilkTide',
  animatedTime: {speed: 'speed'},
  props: {
    colorA: {default: '#0b1026', transform: transformColor},
    colorB: {default: '#ff7e5f', transform: transformColor},
    colorSpace: {default: 'oklch', transform: transformColorSpace, compileTime: true, ui: {type: 'select', options: colorSpaceOptions}},
    highlight: {default: '#fff3dc', transform: transformColor},
    scale: {default: 1.8},
    warp: {default: 0.55},
    lightAngle: {default: 35},
    sheen: {default: 1},
    speed: {default: 1},
  },
  paint: (params) => {
    const {uv} = surfaceOf(params)
    const u = (name: string) => uniformOf(p(name), params)

    // 1 · the field and its slope: three taps of the noise, bound once
    const relief = fdSlope((at) => height(params, at), uv, 0.0035, 'tide')

    // 2 · color: tone-shape the field and read it through the two-color palette
    const shaped = tone(() => relief.value, {glow: 5, contrast: 1.3, balance: 50, invert: true})(params, uv)
    const base = pair(p('colorA'), p('colorB'), p('colorSpace'))(shaped, params)

    // 3 · light: slope → normal → key light → a wrapped diffuse and a sheen
    const tilt = mul(u('sheen'), -0.08)
    const n = nudgeNormal(vec3(0, 0, -1), mul(relief.dx, tilt), mul(relief.dy, tilt), 'tideN')
    const key = keyLightAt(direction(u('lightAngle'), 'tideDir'), -0.7, 'tideKey')
    const diffuse = add(0.55, mul(0.45, lambert(n, key.member('L'), {wrap: 0.5})))
    const spec = dualLobeGlint(dot(n, key.member('H')), {core: [140, 1], halo: [22, 0.15], gain: u('sheen')})

    const rgb = neutralTone(add(mul(base.member('rgb'), diffuse), mul(u('highlight').member('rgb'), spec)))
    return compose.dithered(vec4(rgb, base.member('a')), params)
  },
})
```
