# Blurs, glows & screens

Effects that read the layer inside them (the child) as a texture and rebuild it: blurs, glows,
sharpening, pixelation, shadows, retro screens, glitches, fluted glass, page peels and mirror
floors. A **pointwise** effect edits the child's color at one pixel; the words here are
**gather** effects, which sample the child at other pixels too, so they can smear, spread and
displace it. Every word here goes in a filter definition's `effect:` field, or spreads into a
definition when it needs its own pass over the child first.

Finished effects are one call: `gaussianBlur`, `motionBlur`, `bloom`, `pixelate`, `sharpen`,
`scatter`, `bokehDefocus`. Stacked looks are assembled with `gatherStack` from a **sampling
stage** (how the child is first read: `rgbSplit`, `chromaSmearTaps`, `crispTap`, `childTap`)
and ordered **overlay stages** that edit the result (`scanlines`, `adjust`, `rimLit`,
`shadowComposite`). When several stages need the same geometry, build a **frame** once
(`sphereFrame`, `glitchFrame`, `fluteFrame`, `peelFrame`) and hand it to each of them.

Blurs that need a separate pass (`gaussianBlur`, `channelBlur`, `progressiveBlur`,
`tiltShift`, `bloom`, `bokehDefocus`) return both halves at once. Spread them into a definition
that declares `species: 'custom'`, `requiresRTT: true` and `requiresChild: true` instead of
using `effect:`.

## Reach for it when

| When | Use |
|---|---|
| a soft, even blur | `gaussianBlur` |
| a directional streak, a spin, or a zoom blur | `motionBlur` with `blurPath` |
| a blur that fades in across the canvas | `progressiveBlur` |
| a band in focus with blurred edges | `tiltShift` |
| a glow around bright parts | `bloom` |
| a lens blur with shaped highlight discs | `bokehDefocus` |
| a chromatic fringe from blurring channels apart | `channelBlur` |
| grainy diffusion rather than a smooth blur | `scatter` |
| crisper detail | `sharpen` |
| a pixel grid, with gaps or rounded cells | `pixelate` |
| a soft shadow behind the child | `childTap` + `shadowOffset`, `silhouetteCoverage`, `shadowComposite` |
| an old CRT monitor | `rgbSplit` + `adjust`, `scanlines`, `phosphorMask`, `vignetteOverlay`, `opaque` |
| a worn VHS tape | `chromaSmearTaps` with `tapeWarp`, then `beatPulse` |
| digital glitch bursts | `glitchFrame` + `glitchRgbSplit`, `colorBarFills`, `distortedScanlines` |
| the child bulging on a sphere | `sphereFrame` + `crispTap`, `rimLit` |
| reeded or fluted glass over the child | `fluteFrame` + `refractedTaps`, `fluteHighlight` |
| a page curling up from a corner | `peelFrame` + `peelCompose` with the four peel shades |
| a mirror floor with a fading, softening reflection | `mirrorAcrossRow`, `depthRampBlur`, `planarReflection` |
| film halation or phosphor bloom over a graded color | `screenedBloom` |
| a projector-gate wobble | `frameSway` |

## Order

- gaussianBlur
- motionBlur
- blurPath
- progressiveBlur
- tiltShift
- bloom
- bokehDefocus
- channelBlur
- scatter
- sharpen
- pixelate
- gatherStack
- childTap
- shadowOffset
- silhouetteCoverage
- shadowComposite
- rgbSplit
- adjust
- scanlines
- phosphorMask
- vignetteOverlay
- opaque
- tapeWarp
- chromaSmearTaps
- beatPulse
- glitchFrame
- glitchRgbSplit
- colorBarFills
- distortedScanlines
- sphereFrame
- crispTap
- rimLit
- fluteFrame
- refractedTaps
- fluteHighlight
- peelFrame
- peelCompose
- curlShading
- curlSheen
- overhangShadow
- revealShadow
- mirrorAcrossRow
- depthRampBlur
- planarReflection
- screenedBloom
- frameSway

## Example

```ts
import {defineShader, p, effects, transformColor, transformBoolean} from 'shaders/std'

const {gatherStack, childTap, shadowOffset, silhouetteCoverage, shadowComposite} = effects.blurs

// A soft drop shadow: read the child's silhouette offset in a compass direction, blur it,
// tint it, and place it behind the child.
export const SoftShadow = defineShader({
  name: 'SoftShadow',
  props: {
    color: {default: '#000000', transform: transformColor},
    distance: {default: 0.1},
    angle: {default: 135},
    blur: {default: 5},
    intensity: {default: 0.5},
    cutout: {default: false, transform: transformBoolean, compileTime: true},
  },
  effect: gatherStack(childTap(), [
    shadowComposite({
      coverage: silhouetteCoverage({
        at: shadowOffset({angle: p('angle'), distance: p('distance')}),
        blur: p('blur'),
      }),
      color: p('color'),
      intensity: p('intensity'),
      cutout: p('cutout'),
    }),
  ]),
})
```
