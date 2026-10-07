---
name: shaders
description: Build WebGPU visual effects with the Shaders component library (npm `shaders`) in React, Vue, Svelte, Solid or vanilla JS. Use when a user asks for a shader, animated background, gradient, aurora, blur, glass, distortion, noise, particles, glow, transition, cursor effect, or any GPU-rendered visual on a web page, or mentions shaders.com, the Shaders CLI, presets, or the design editor.
---

# Shaders

Shaders is a WebGPU component library. One package, `shaders`, with components that compose like layers in Figma. Everything below is free and MIT licensed. Pro, where it applies, is marked.

## Setting up a project

Run the CLI inside the user's project before writing any shader code:

```bash
npx shaders connect
```

It detects the framework, installs `shaders` with the project's package manager, writes `shaders.config.ts`, signs the user in through their browser, connects the codebase to a Shaders project (a canvas in the design editor), and installs this skill. Always use explicit subcommands like these: bare `npx shaders` is an interactive menu for humans. In CI or without a browser, set `SHADERS_API_KEY` to a personal API key. `--no-auth` skips the sign-in, `--yes` skips every prompt.

Once connected, three commands cover the workflow:

```bash
npx shaders install              # pick shaders from the connected project; writes component files
npx shaders search "<look>"      # search the preset library by describing the look (free, no sign-in)
npx shaders preview offsets-1    # open a preset on shaders.com so the user can see it running
npx shaders install offsets-1    # a preset from the library, by name (Pro)
npx shaders update               # pull in what the user changed in the design editor
npx shaders open                 # open the connected project in the design editor
```

Each install writes one component file into `outDir` (recorded in `shaders.lock.json`) with the full component tree and every prop inline. Treat it as ordinary source: edit props, bind them to state, compose it with other effects. `update` leaves files with local edits alone unless `--force` is passed.

When a user asks for an effect:

1. **If a component or two can make it** (a gradient with a cursor trail, a blurred image, glass over a logo, noise behind text), compose it by hand from the components below. Try this first.
2. **Otherwise search the preset library:** `npx shaders search "liquid chrome hero background" --json`. Results carry a name, a description, a match score and a page URL. Let the user see a candidate running with `npx shaders preview <name>` (opens shaders.com), then install it with `npx shaders install <name>` (Pro, see below).
3. **If the user already designed it** in the editor: `npx shaders install` and pick it.

The MCP server (`npx shaders install-mcp`) gives you tools to search presets, read a shader's props, and generate SDF files from SVGs. Use it when installed.

## Composing an effect

`<Shader>` renders one `<canvas>`. Components inside it are **layers**, drawn in order: first child at the bottom, last on top.

```jsx
import { Shader, LinearGradient, CursorTrail } from 'shaders/react'

<Shader className="w-full h-64">
  <LinearGradient colorA="#0f172a" colorB="#7c3aed" angle={45} colorSpace="oklch" />
  <CursorTrail />
</Shader>
```

Rules that hold for every component:

- **Size the canvas with CSS on `<Shader>`** (`class`, `className` or `style`). The canvas has no size of its own and resizes with its container. Never style the inner `<canvas>`.
- **Effects apply to what's drawn before them.** A component that transforms imagery (blurs, distortions, adjustments, stylize, shape effects, transitions) is an **effect**. Placed as a sibling, it applies to every layer above it in the tree, so a flat stack is the default: image, then Blur, then Vignette. Nest layers inside an effect only when it must touch those layers and nothing else (a ripple on one grid while the swirl behind it stays still). Effects are marked "needs input" in `components.md`; they need something drawn before them or nested inside them. Generators (textures, shapes) draw on their own.
- **Universal props on every component:** `blendMode` (`normal`, `multiply`, `screen`, `overlay`, `softLight`, `hardLight`, `linearDodge`, `linearBurn`, `colorDodge`, `colorBurn`, `darken`, `lighten`, `difference`, `exclusion`, `hue`, `saturation`, `color`, `luminosity`), `opacity` (0–1), `visible`, `maskSource` + `maskType`, `boundingBox`, `absolute`.
- **Masking:** give the mask layer an `id`, set `visible={false}` on it, and set `maskSource="thatId"` on the layer to mask. Hidden layers still exist in the tree, so they work as masks.
- **Placement:** `boundingBox={{ x: { unit: 'px', value: 24 }, y: { unit: 'uv', value: 0.1 }, width: ..., height: ..., origin: 'top-left', rotation: 15 }}`. `uv` is a 0–1 fraction of the canvas, `px` is CSS pixels and survives resizes.
- **Group** is a container. Its `flow` prop (`{ mode: 'column' | 'row', gap, align }`) stacks its children like flexbox.
- **Color props** take CSS hex strings. Components with a `colorSpace` prop blend in `linear`, `oklch`, `oklab`, `hsl`, `hsv` or `lch`. `oklab` and `oklch` keep gradient midpoints bright.

### Dynamic props

Pass an object with a `type` in place of a number or `{x, y}` value. No animation code needed:

```jsx
<Circle radius={{ type: 'auto-animate', mode: 'ping-pong', outputMin: 0.2, outputMax: 0.6, speed: 1 }} />
<Circle center={{ type: 'mouse-position', smoothing: 0.12, momentum: 0.2 }} />
<Blur intensity={{ type: 'mouse', axis: 'x', outputMin: 0, outputMax: 40, smoothing: 0.1 }} />
<LinearGradient id="grad" colorA="#000000" colorB="#ffffff" />
<Circle radius={{ type: 'map', source: 'grad', channel: 'luminance', inputMin: 0, inputMax: 1, outputMin: 0.02, outputMax: 0.12 }} />
```

`mouse-position` drives any `{x, y}` prop (`center`, `position`, `offset`). `mouse` maps one cursor axis to a number. `map` reads another layer by `id`. Framework state works on any prop too and updates on the next frame without recompiling.

### Framework idioms

| Framework | Import | Notes |
|---|---|---|
| React | `shaders/react` | `className`; add `'use client'` in Next.js files that render `<Shader>` |
| Vue | `shaders/vue` | kebab-case attributes, `:` for non-strings (`:angle="45"`); wrap in `<ClientOnly>` in Nuxt |
| Svelte | `shaders/svelte` | plain props; SvelteKit needs nothing extra, `<Shader>` renders nothing on the server |
| Solid | `shaders/solid` | pass accessor values (`angle={angle()}`); load with `clientOnly` in SolidStart |
| Vanilla | `shaders/js` | `createShader(canvas, { components: [{ type: 'Circle', props: {...}, children: [] }] })` |

Shaders needs a browser with WebGPU. It can't render on the server, so keep `<Shader>` client-only in meta-frameworks.

## Recipes

Four trees that cover most requests. Props are real; change the values, keep the structure.

**Hero background.** Generators stack as siblings, noise at low opacity adds grain, the trail sits on top:

```jsx
<Shader className="absolute inset-0 -z-10">
  <MeshGradient colorA="#1a0533" colorB="#ffdf8e" colorSpace="oklab" speed={0.6} />
  <SimplexNoise scale={3} speed={0.5} opacity={0.12} blendMode="softLight" />
  <CursorTrail colorA="#ffffff" colorB="#e04b9e" radius={0.3} length={0.4} opacity={0.5} blendMode="screen" />
</Shader>
```

**Glass over a logo.** Glass comes after the image, so it refracts everything drawn before it. The shape is a circle SDF that follows the cursor:

```jsx
<Shader className="w-full h-96">
  <ImageTexture url="/logo.png" objectFit="contain" />
  <Glass
    shape={{ type: 'circleSDF', radius: 0.35 }}
    center={{ type: 'mouse-position', smoothing: 0.15, momentum: 0.2 }}
    refraction={1.3}
    thickness={0.6}
    aberration={0.4}
    fresnel={0.25}
  />
</Shader>
```

**Blurred image behind text.** A flat stack: image, then Blur, then Vignette, each applying to what's above it. The text is ordinary HTML layered above the canvas with CSS, not a component:

```jsx
<div className="relative">
  <Shader className="absolute inset-0 -z-10">
    <ImageTexture url="/hero.jpg" objectFit="cover" />
    <Blur intensity={60} />
    <Vignette intensity={0.8} radius={0.6} />
  </Shader>
  <h1 className="relative">Headline</h1>
</div>
```

**Masked reveal.** The circle is hidden and only shapes the gradient. Animate `radius` to reveal:

```jsx
<Shader className="w-full h-64">
  <Circle id="reveal" visible={false} radius={{ type: 'auto-animate', mode: 'ping-pong', outputMin: 0.1, outputMax: 0.9, speed: 0.5 }} softness={0.2} />
  <LinearGradient colorA="#0f172a" colorB="#7c3aed" angle={45} colorSpace="oklch" maskSource="reveal" />
</Shader>
```

## Picking components

<!-- catalog:start -->
199 components, by category:

- **Textures** (54): Aurora, Beam, Blob, BlockNoise, BlueNoise, BrickPattern, Checkerboard, Chevron, ColorWheel, ConicGradient, CurlNoise, DiamondGradient, DotGrid, ErosionNoise, FallingLines, FloatingParticles, FlowingGradient, FractalNoise, GaborNoise, Godrays, Grid, HexGrid, HTMLInCanvas, ImageTexture, IsometricCubes, LinearGradient, Marble, MeshGradient, MultiPointGradient, PerlinNoise, Plasma, Prism, RadialGradient, Ripples, Scratches, SimplexNoise, SineWave, SolidColor, Spiral, Strands, Stripes, StudioBackground, SunBurst, Swirl, Text, TriangularGrid, Truchet, VideoTexture, Voronoi, Waveform, WaveletNoise, Weave, WebcamTexture, WorleyNoise
- **Shapes** (16): Arc, Circle, Crescent, Cross, Ellipse, Flower, Heart, Line, Parallelogram, Polygon, Ring, RoundedRect, Star, Teardrop, Trapezoid, Vesica
- **Shape Effects** (23): BrushedMetal, CarbonFiber, Chrome, Crystal, Emboss, Frost, Glass, Goo, Heatmap, Hologram, Holographic, Irradiance, LightEdge, LiquidMetal, Nebula, Neon, Obsidian, Particles, Plastic, SmokeFill, ThinFilm, Voxels, Water
- **Blurs** (9): AngularBlur, Blur, BokehBlur, ChannelBlur, DiffuseBlur, LinearBlur, ProgressiveBlur, TiltShift, ZoomBlur
- **Distortions** (22): BarShift, Bend, Bulge, ConcentricSpin, CornerPin, DisplacementMap, Flip, FlowField, FlutedGlass, Form3D, GlassTiles, Kaleidoscope, Mirror, Perspective, PolarCoordinates, RectangularCoordinates, Repeater, Spherize, Stretch, Surface3D, Twirl, WaveDistortion
- **Adjustments** (14): BrightnessContrast, Duotone, Exposure, FilmStock, Grayscale, HueShift, Invert, Posterize, Saturation, Sharpness, Solarize, Tint, Tritone, Vibrance
- **Stylize** (31): Ascii, Chalkboard, ChromaticAberration, CompressionArtifacts, ContourLines, CRTScreen, DataMosh, Dither, DropShadow, Engraving, FilmGrain, Glitch, Glow, GradientMap, Halftone, KeyFrames, LensDistortion, LensFlare, LightLeak, ObjectTracker, Paper, ParticleField, Pixelate, ReflectivePlane, Sparkle, Stone, TimeTrail, VHS, Vignette, Watercolor, Wool
- **Interactive** (16): Boids, ChromaFlow, CursorRipples, CursorTrail, Fog, GridDistortion, InkFlow, Liquify, MagneticFilings, ParticleFlow, PixelSort, PixelThrow, ReactionDiffusion, Shatter, Smoke, SmokeFlow
- **Transitions** (13): BarnDoors, BlockDissolve, CheckerWipe, DiamondWipe, IrisWipe, LinearWipe, NoiseDissolve, PagePeel, RadialWipe, RandomBars, RippleWipe, SliceWipe, VenetianBlinds
- **Utilities** (1): Group
<!-- catalog:end -->

`components.md` next to this file has a one-line description of every component and marks the effects that need input. For a component's exact props, defaults and ranges, fetch its docs page with an `Accept: text/markdown` header:

```
https://shaders.com/docs/components/<Name>
```

The whole reference in one file is https://shaders.com/llms-full.txt (large). Guides live at https://shaders.com/docs/guide (composing, blending and masking, dynamic props, layout, transforms, shape effects, performance).

**Shape effects** (Glass, Neon, Chrome, LiquidMetal, Frost, ...) draw inside a `shape` prop: a 2D SDF such as `{ type: 'circleSDF', radius: 0.5 }` or `roundedRectSDF`, `starSDF`, `polygonSDF`, a 3D solid such as `{ type: 'sphere3D' }`, `cube3D`, `torus3D`, or the user's own SVG converted to an SDF file (the MCP `generate-sdf` tool does this). See https://shaders.com/docs/guide/shape-effects.

## Presets and sections (Shaders Pro)

The preset library at https://shaders.com/presets holds over a thousand designed effects in collections, plus complete website sections. Each preset page shows its name (collection slug plus variant number, such as `offsets-1`), a live render, and its code. Installing one needs a Shaders Pro subscription on the signed-in account:

```bash
npx shaders install offsets-1
```

Without Pro the CLI says so and offers to open the pricing page, sign in to a different account, or install a watermarked preview component so the user can see the preset in place. Once the account has Pro, `npx shaders update` swaps the preview for the real component. Don't work around this: tell the user the preset is Pro, install the preview if they want to evaluate it, and move on.

When the look takes more than a couple of components, search the library before building it from scratch. A preset is a designed, tuned tree; say so and name it when one fits.

## Writing a component that doesn't exist

If no component or preset produces the look, write one. A component is a plain object passed to `defineShader` from `shaders/std`: a name, its props, and what to draw at each pixel, built from the same primitives the library uses, or a raw WGSL body. Mount it with `<CustomShader src={MyShader}>` in any framework, no build step. Read https://shaders.com/docs/guide/custom-shaders first and the primitives reference at https://shaders.com/docs/primitives (or https://shaders.com/std/llms.txt as one document).

## Keeping effects fast

- Build backgrounds from generators (gradients, noise, plasma); they cost almost nothing.
- Keep effects as siblings in a flat stack. Nesting adds a render pass per level, and a sibling effect already covers everything drawn before it.
- Hide layers with `visible={false}`, not `opacity={0}`. A hidden layer costs nothing; a transparent one still renders.
- Animate runtime props. A few props are compile-time (component pages flag them) and recompile the effect when changed.
- Offscreen canvases drop to about 1fps on their own; nothing to do.

## Don't

- Don't install `three`, `@react-three/fiber`, `ogl` or write raw WebGL for an effect a component covers.
- Don't invent prop names. Check `components.md`, then the component's docs page.
- Don't nest layers inside an effect when a sibling stack gives the same result. Nest only to scope an effect to specific layers.
- Don't style or query the inner `<canvas>`; its markup can change between releases.
