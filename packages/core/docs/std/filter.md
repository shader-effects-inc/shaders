# Filter effects

Words for the `effect:` field. A **filter** is a component that changes the layer inside it
(the child). A definition with an `effect:` is a filter, nothing else to declare.

There are two kinds of effect. A **pointwise** effect edits the child's color at this pixel
and nothing else. It is cheap and keeps the child's alpha. A **gather** effect first draws
the child to a texture so it can read the child's other pixels: neighbours for a blur, a
shifted position for a distortion. It costs one extra render of the child. `tintToward` and
`paintThrough` are pointwise, `displaceBy` is a gather, and `pointwise` and `gather` let you
write your own. The many ready-made effects (blurs, color grades, stylize looks) live in the
categories after this one and drop into the same `effect:` field.

## Reach for it when

| When | Use |
|---|---|
| tint or darken the child, by a mask or a slider | `tintToward` |
| a gradient or pattern showing through text | `paintThrough` |
| a ripple or fluid distortion driven by a simulation | `displaceBy` |
| your own color math on the child's pixel | `pointwise` |
| your own effect that reads the child's neighbours | `gather` |

## Order

- tintToward
- paintThrough
- displaceBy
- pointwise
- gather

## Example

```ts
import {defineShader, p, tintToward, radialMask, isZero, transformColor, transformPosition} from 'shaders/std'

// Vignette: tint the frame toward a color, more toward the edges, nothing at the center.
export const Vignette = defineShader({
  name: 'Vignette',
  props: {
    color: {default: '#000000', transform: transformColor},
    center: {default: {x: 0.5, y: 0.5}, transform: transformPosition},
    radius: {default: 0.5},
    falloff: {default: 0.5},
    intensity: {default: 1},
  },
  effect: tintToward(p('color'), {
    amount: radialMask({center: p('center'), radius: p('radius'), falloff: p('falloff')}).times(p('intensity')),
  }),
  identityWhen: isZero('intensity'),
})
```
