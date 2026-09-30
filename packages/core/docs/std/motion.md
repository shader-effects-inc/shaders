# Motion & timing

Words that turn a clock into rhythm. The clock is a number of seconds: declare
`animatedTime: {speed: 'speed'}` on the definition and read the layer's clock with
`animatedTime(params)`, and a speed prop and a pause button come for free. `cycle` splits
that clock into "which pass is this" and "how far through it are we", `cycleSeed` gives each
pass a stable random number, `oscillate` and `pulseTrain` shape it into a sweep or a blink,
and the easings bend a 0..1 progress. The result is a number you multiply into a color, a
radius or a coordinate.

`oscillating` and `pulsing` are the same shapes packaged as a value you can drop into any
slot that takes a prop, on a warp or an effect, so a knob moves on its own with no math in
your definition.

## Reach for it when

| When | Use |
|---|---|
| a warp or effect knob that sweeps back and forth by itself | `oscillating` in the slot |
| a knob that switches on and off in a rhythm | `pulsing` in the slot |
| something happens every N seconds | `cycle`, which gives progress and index |
| each pass should look different but hold still while it lasts | `cycleSeed` on the cycle's index |
| a smooth breathing value between two numbers | `oscillate` |
| a blink or strobe with a controllable on time | `pulseTrain` |
| just the 0..1 position inside a repeating cycle | `phase` |
| a progress that accelerates, settles or glides | `easeIn`, `easeOut`, `easeInOut` |

## Order

- oscillating
- pulsing
- cycle
- cycleSeed
- oscillate
- pulseTrain
- phase
- easeInOut
- easeOut
- easeIn

## Example

```ts
import {defineShader, p, uniformOf, animatedTime, math, motion, transformColor} from 'shaders/std'

const {sub, mul, mix, splat3, vec4} = math

// Flash: every `period` seconds the canvas lights up in a new random blend of two colors and fades out.
export const Flash = defineShader({
  name: 'Flash',
  animatedTime: {speed: 'speed'},
  props: {
    colorA: {default: '#ff7e5f', transform: transformColor},
    colorB: {default: '#7dd3fc', transform: transformColor},
    period: {default: 2},
    speed: {default: 1},
  },
  paint: (params) => {
    const u = (name: string) => uniformOf(p(name), params)
    const t = animatedTime(params)
    const {progress, index} = motion.cycle(t, u('period'))
    const fade = motion.easeOut(sub(1, progress)) // bright at the start of each pass, then out
    const pick = motion.cycleSeed(index) // one stable random blend per pass
    const breathe = motion.oscillate(t, {rate: 0.5, min: 0.7, max: 1})
    const rgb = mix(u('colorA').member('rgb'), u('colorB').member('rgb'), splat3(pick))
    return vec4(mul(rgb, mul(fade, breathe)), 1)
  },
})
```
