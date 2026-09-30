# Gradients

Finished gradient paints: a flat fill, a color wheel that cycles over time, a cloud of five
colored points, and a beam of light between two points. Each word binds your props and
returns a paint, so a definition is the props plus one line. They are complete looks rather
than building blocks. For a gradient of your own shape, compose a `dist` field with `rampOver`
from the fields words instead.

Colors mix in the color space you declare on a `compileTime` prop, and positions are props
made with `transformPosition`. The result goes in the definition's `paint:` field.

## Reach for it when

| When | Use |
|---|---|
| a single color, possibly translucent | `solidColor` |
| a rainbow or three-color cycle that drifts across the canvas | `colorWheel` |
| five draggable color points blended by nearness | `pointCloudGradient` |
| a soft beam of light from one point to another | `beam` with `beamPreconvertedFields` |

## Order

- solidColor
- beam
- beamPreconvertedFields
- colorWheel
- pointCloudGradient

## Example

```ts
import {defineShader, p, gradients, transformColor, transformPosition, transformColorSpace, colorSpaceOptions} from 'shaders/std'

// A tapered beam: wide and soft at the start, tight at the end, red core fading to blue.
export const Beam = defineShader({
  name: 'Beam',
  extraFields: gradients.beamPreconvertedFields,
  props: {
    startPosition: {default: {x: 0.2, y: 0.5}, transform: transformPosition},
    endPosition: {default: {x: 0.8, y: 0.5}, transform: transformPosition},
    startThickness: {default: 0.4},
    endThickness: {default: 0.1},
    startSoftness: {default: 1},
    endSoftness: {default: 0.3},
    insideColor: {default: '#ff3b30', transform: transformColor},
    outsideColor: {default: '#1e40ff', transform: transformColor},
    colorSpace: {default: 'oklab', transform: transformColorSpace, compileTime: true, ui: {type: 'select', options: colorSpaceOptions}},
  },
  paint: gradients.beam({
    from: p('startPosition'),
    to: p('endPosition'),
    thickness: {start: p('startThickness'), end: p('endThickness')},
    softness: {start: p('startSoftness'), end: p('endSoftness')},
    colors: {inside: p('insideColor'), outside: p('outsideColor')},
    space: p('colorSpace'),
  }),
})
```
