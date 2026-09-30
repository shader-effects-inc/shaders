# Figures & strokes

Drawn lines: a straight segment and a traveling sine wave. Each word binds your props and
returns a paint that draws the stroke in one color prop with an antialiased edge and leaves
everything else transparent, so it layers cleanly over what sits behind it.

Lengths are fractions of the canvas height, so a stroke keeps its weight when the canvas
resizes. Endpoints and centers are position props made with `transformPosition`. The result
goes in the definition's `paint:` field.

## Reach for it when

| When | Use |
|---|---|
| a line between two draggable points, solid, dashed or dotted | `strokedSegment` |
| a wavy line that ripples along itself | `sineStroke` |

## Order

- strokedSegment
- sineStroke

## Example

```ts
import {defineShader, p, figures, transformColor, transformPosition} from 'shaders/std'

// A dashed line with round ends. The style select maps its labels to the numbers the word takes.
export const DashedLine = defineShader({
  name: 'DashedLine',
  props: {
    color: {default: '#ffffff', transform: transformColor},
    pointA: {default: {x: 0.2, y: 0.5}, transform: transformPosition},
    pointB: {default: {x: 0.8, y: 0.5}, transform: transformPosition},
    thickness: {default: 0.01, ui: {type: 'range', min: 0, max: 0.25, step: 0.001}},
    style: {
      default: 'dashed',
      transform: (value: string) => ({solid: 0, dashed: 1, dotted: 2} as Record<string, number>)[value] ?? 0,
      ui: {type: 'select', options: [{label: 'Solid', value: 'solid'}, {label: 'Dashed', value: 'dashed'}, {label: 'Dotted', value: 'dotted'}]},
    },
    dashLength: {default: 0.05},
    gapLength: {default: 0.025},
  },
  paint: figures.strokedSegment({
    from: p('pointA'),
    to: p('pointB'),
    width: p('thickness'),
    style: p('style'),
    dashLength: p('dashLength'),
    gapLength: p('gapLength'),
    capStart: 1,
    capEnd: 1,
    color: p('color'),
  }),
})
```
