# Props & values

How a definition points at its own props. `p('radius')` is a **prop ref**: hand it to any word
that takes a color, a position or a number and the word reads that prop live on the GPU, with
drivers and animation applied, and no rebuild when the value changes. A **Scalar** is a number
built from props and masks. `radialMask(…)` returns one, so does `oscillating(…)` from
`motion`, and `.times(p('intensity'))` scales one by a prop. Words that take a `ScalarInput`
accept either a prop ref or a Scalar.

Prop refs and Scalars are data, not GPU values. A word turns them into GPU values when the
shader is built. In a hand-written builder you do that yourself with `uniformOf` and
`resolveScalar` from the invocation words.

## Reach for it when

| When | Use |
|---|---|
| a word should read one of your props | `p` |
| a strength that fades with distance from a point, scaled by a prop | `radialMask` chained with .times(p('intensity')) on the `Scalar` |
| a value that breathes or pulses on its own | `oscillating` or `pulsing` from motion, as a `Scalar` |
| a prop's value inside a hand-written builder | `uniformOf` |

## Order

- p
- Scalar

## Example

```ts
import {defineShader, p, tintToward, radialMask, crosses, isZero, transformColor, transformPosition} from 'shaders/std'

// A vignette: tint the layer inside toward a color, strongest past a radius from a point.
export const Vignette = defineShader({
  name: 'Vignette',
  props: {
    color: {default: '#000000', transform: transformColor},
    center: {default: {x: 0.5, y: 0.5}, transform: transformPosition},
    radius: {default: 0.4},
    falloff: {default: 0.6},
    intensity: {default: 1, recompile: crosses(0)},
  },
  effect: tintToward(p('color'), {
    amount: radialMask({center: p('center'), radius: p('radius'), falloff: p('falloff')}).times(p('intensity')),
  }),
  identityWhen: isZero('intensity'),
  missingChildMessage: 'Nest a layer inside Vignette to darken its edges.',
})
```
