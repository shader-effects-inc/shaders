/**
 * Motion: easing curves, loops, per-element timing and envelopes. Pure functions of time, so a
 * thousand elements can each run their own animation in one pass. Times are in whatever unit you
 * pass (the `time` built-in is seconds).
 */
import {libFn} from '../wgslLibrary'
import {hash11} from './noise'

export const easeIn = libFn({
    name: 'easeIn',
    family: 'motion',
    signature: 'easeIn(t: f32) -> f32  // t 0..1 (clamped) -> 0..1, cubic',
    doc: 'Starts slow and speeds up (cubic).',
    code: `
fn easeIn(t: f32) -> f32 {
    let x = clamp(t, 0.0, 1.0);
    return x * x * x;
}
`,
})

export const easeOut = libFn({
    name: 'easeOut',
    family: 'motion',
    signature: 'easeOut(t: f32) -> f32  // t 0..1 (clamped) -> 0..1, cubic',
    doc: 'Starts fast and settles gently (cubic). The natural curve for things arriving.',
    code: `
fn easeOut(t: f32) -> f32 {
    let y = 1.0 - clamp(t, 0.0, 1.0);
    return 1.0 - y * y * y;
}
`,
})

export const easeInOut = libFn({
    name: 'easeInOut',
    family: 'motion',
    signature: 'easeInOut(t: f32) -> f32  // t 0..1 (clamped) -> 0..1, cubic',
    doc: 'Slow at both ends, fastest in the middle (cubic). The natural curve for things moving from one place to another.',
    code: `
fn easeInOut(t: f32) -> f32 {
    let x = clamp(t, 0.0, 1.0);
    let y = 2.0 - 2.0 * x;
    return select(1.0 - y * y * y * 0.5, 4.0 * x * x * x, x < 0.5);
}
`,
})

export const pingPong = libFn({
    name: 'pingPong',
    family: 'motion',
    signature: 'pingPong(t: f32, period: f32) -> f32  // 0..1..0 once per period',
    doc: 'Goes 0 to 1 and back to 0 every period, linearly (0 at t = 0, 1 at half the period). Wrap it in easeInOut for a soft back-and-forth.',
    code: `
fn pingPong(t: f32, period: f32) -> f32 {
    return 1.0 - abs(1.0 - 2.0 * fract(t / max(period, 1e-6)));
}
`,
})

export const cycle = libFn({
    name: 'cycle',
    family: 'motion',
    signature: 'cycle(time: f32, period: f32) -> Cycle { phase: f32 /* 0..1 */, seed: f32 /* 0..1, new each cycle */, index: f32 /* cycle count */ }',
    doc: 'Split time into repeating cycles: phase runs 0 to 1 in each, seed is a fresh random number per cycle (to vary each repeat), index counts the cycles.',
    deps: [hash11],
    code: `
struct Cycle { phase: f32, seed: f32, index: f32 }
fn cycle(time: f32, period: f32) -> Cycle {
    let x = time / max(period, 1e-6);
    let n = floor(x);
    return Cycle(x - n, hash11(n), n);
}
`,
})

export const stagger = libFn({
    name: 'stagger',
    family: 'motion',
    signature: 'stagger(index: f32, count: f32, order: i32) -> f32  // 0..1 delay; order 0 in order, 1 reverse, 2 random, 3 centre-out',
    doc: 'A start delay (0..1) for element index out of count, so a group animates one after another: 0 first to last, 1 last to first, 2 random order, 3 from the middle outward. Typical use: easeOut(clamp(t * 2.0 - stagger(i, n, 0), 0.0, 1.0)).',
    deps: [hash11],
    code: `
fn stagger(index: f32, count: f32, order: i32) -> f32 {
    let last = max(count - 1.0, 1.0);
    let lin = clamp(index / last, 0.0, 1.0);
    let mid = last * 0.5;
    var r = lin;
    if (order == 1) {
        r = 1.0 - lin;
    } else if (order == 2) {
        r = hash11(index);
    } else if (order == 3) {
        r = clamp(abs(index - mid) / mid, 0.0, 1.0);
    }
    return select(r, 0.0, count <= 1.0);
}
`,
})

export const springStep = libFn({
    name: 'springStep',
    family: 'motion',
    signature: 'springStep(t: f32, stiffness: f32, damping: f32) -> f32  // 0 at t <= 0, settles at 1; may overshoot',
    doc: 'A spring released toward 1 at t = 0 (mass 1, exact solution): low damping overshoots and wobbles, damping >= 2 * sqrt(stiffness) glides in without overshoot. stiffness 100, damping 10 is a lively default with t in seconds.',
    code: `
fn springStep(t: f32, stiffness: f32, damping: f32) -> f32 {
    let w0 = sqrt(max(stiffness, 1e-4));
    let z = max(damping, 0.0) / (2.0 * w0);
    let tt = max(t, 0.0);
    var x: f32;
    if (z < 1.0) {
        let wd = w0 * sqrt(1.0 - z * z);
        x = 1.0 - exp(-z * w0 * tt) * (cos(wd * tt) + (z * w0 / wd) * sin(wd * tt));
    } else if (z < 1.001) {
        x = 1.0 - exp(-w0 * tt) * (1.0 + w0 * tt);
    } else {
        let s = sqrt(z * z - 1.0);
        let r1 = -w0 * (z - s);
        let r2 = -w0 * (z + s);
        x = 1.0 - (r2 * exp(r1 * tt) - r1 * exp(r2 * tt)) / (r2 - r1);
    }
    return x;
}
`,
})

export const impulse = libFn({
    name: 'impulse',
    family: 'motion',
    signature: 'impulse(t: f32, attack: f32, decay: f32) -> f32  // 0..1, peak 1 at t = attack',
    doc: 'A hit: 0 before t = 0, rises smoothly to 1 over attack, then fades exponentially (to about a third after decay more). For flashes, ripples and anything triggered.',
    code: `
fn impulse(t: f32, attack: f32, decay: f32) -> f32 {
    let a = max(attack, 1e-5);
    let x = clamp(t / a, 0.0, 1.0);
    let rise = x * x * (3.0 - 2.0 * x);
    let fall = exp(-max(t - a, 0.0) / max(decay, 1e-5));
    return select(select(fall, rise, t < a), 0.0, t < 0.0);
}
`,
})

export const pulse = libFn({
    name: 'pulse',
    family: 'motion',
    signature: 'pulse(t: f32, shape: i32) -> f32  // 0..1, period 1 in t; shape 0 sine, 1 triangle, 2 heartbeat',
    doc: 'A repeating beat, one per unit of t (multiply time by a rate): 0 a smooth sine swell, 1 a linear triangle, 2 a heartbeat (a strong beat, a softer second beat, then rest). Each starts at about 0.',
    code: `
fn pulse(t: f32, shape: i32) -> f32 {
    let f = fract(t);
    let sine = 0.5 - 0.5 * cos(f * 6.28318531);
    let tri = 1.0 - abs(2.0 * f - 1.0);
    let b1 = (f - 0.1) / 0.045;
    let b2 = (f - 0.32) / 0.06;
    let heart = min(exp(-b1 * b1) + 0.6 * exp(-b2 * b2), 1.0);
    return select(select(sine, tri, shape == 1), heart, shape == 2);
}
`,
})

export const jitterHold = libFn({
    name: 'jitterHold',
    family: 'motion',
    signature: 'jitterHold(t: f32, rate: f32, hold: f32) -> f32  // 0..1; rate = new values per unit t, hold 0..1',
    doc: 'A random value that jumps rate times per unit of t and holds in between: hold 1 snaps (sample and hold, glitchy), lower values glide to each new value over the last (1 - hold) of the step.',
    deps: [hash11],
    code: `
fn jitterHold(t: f32, rate: f32, hold: f32) -> f32 {
    let x = t * rate;
    let n = floor(x);
    let f = x - n;
    let h = clamp(hold, 0.0, 1.0);
    let m = clamp((f - h) / max(1.0 - h, 1e-5), 0.0, 1.0);
    return mix(hash11(n), hash11(n + 1.0), m * m * (3.0 - 2.0 * m));
}
`,
})
