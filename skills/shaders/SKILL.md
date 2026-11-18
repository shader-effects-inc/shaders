---
name: shaders
description: Write WebGPU effects in code with the open-source Shaders engine (npm `shaders`) in React, Vue, Svelte, Solid or vanilla JS. Use when building or editing Shaders components in a codebase (composing layers, props, masking, dynamic props, shape effects, or a custom component), or when editing a shader installed from the Shaders app. When the Shaders MCP is connected, design on its canvas instead and follow its guides.
---

# Shaders

Shaders is a WebGPU component library. One package, `shaders`, with components that compose like layers in Figma. Everything below is free and MIT licensed.

This skill is for writing Shaders code. It's the reference when there's no canvas to design on, or when you're editing a shader that's already in the codebase.

## Designing with the Shaders MCP

If the Shaders MCP is connected (its tools start with `editor-`), design shaders there, not in code. It builds on a live canvas in the Shaders desktop app where the user watches, you can screenshot and review your work, and its own guides (`editor-guide`, `install-guide`) are always current. Follow them over this file wherever they differ. Come back here once a shader is installed and you're editing its code.

If it isn't connected and the user wants to design a shader, suggest it once: `npx shaders mcp` adds the MCP to their agents, and the app is at https://shaders.com/download. If they'd rather stay in code, carry on with this skill. Writing code, you can't see what you render, so keep compositions simple and ask the user to check the result.

## Setting up a project

Install the package with the project's package manager:

```bash
npm install shaders
```

Shaders installed from the app are component files listed in `shaders.lock.json`, next to `shaders.config.ts`. Treat them as ordinary source: edit props, bind them to state, compose them with other effects. A `<Component>.animations.ts` file beside one holds its keyframe animations; build them with the project's animation library. `npx shaders update` pulls in changes made in the app and skips files with local edits unless `--force` is passed.

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
201 components, by category:

- **Textures** (55): Aurora, Beam, Blob, BlockNoise, BlueNoise, BrickPattern, Checkerboard, Chevron, ColorWheel, ConicGradient, CurlNoise, DiamondGradient, DotGrid, ErosionNoise, FallingLines, FloatingParticles, FlowingGradient, FractalNoise, GaborNoise, Godrays, Grid, HexGrid, HTMLInCanvas, ImageTexture, IsometricCubes, LinearGradient, Marble, MeshGradient, MultiPointGradient, PerlinNoise, Plasma, Prism, RadialGradient, Ripples, Satin, Scratches, SimplexNoise, SineWave, SolidColor, Spiral, Strands, Stripes, StudioBackground, SunBurst, Swirl, Text, TriangularGrid, Truchet, VideoTexture, Voronoi, Waveform, WaveletNoise, Weave, WebcamTexture, WorleyNoise
- **Shapes** (16): Arc, Circle, Crescent, Cross, Ellipse, Flower, Heart, Line, Parallelogram, Polygon, Ring, RoundedRect, Star, Teardrop, Trapezoid, Vesica
- **Shape Effects** (24): BrushedMetal, CarbonFiber, Chrome, Crystal, Emboss, Frost, Glass, Goo, Heatmap, Hologram, Holographic, Irradiance, LightEdge, LiquidMetal, Nebula, Neon, Obsidian, Particles, Plastic, PrismaticGlass, SmokeFill, ThinFilm, Voxels, Water
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

**Shape effects** (Glass, Neon, Chrome, LiquidMetal, Frost, ...) draw inside a `shape` prop: a 2D SDF such as `{ type: 'circleSDF', radius: 0.5 }` or `roundedRectSDF`, `starSDF`, `polygonSDF`, a 3D solid such as `{ type: 'sphere3D' }`, `cube3D`, `torus3D`, or the user's own SVG as `{ type: 'svg', svgUrl: '/logo.svg' }` (built in the browser when the shader loads; the SVG must be same-origin or allow CORS). See https://shaders.com/docs/guide/shape-effects.

## Writing a component that doesn't exist

If no component produces the look, write one. A component is a plain object passed to `defineShader` from `shaders/std`: a name, its props, and what to draw at each pixel, built from the same primitives the library uses, or a raw WGSL body. Mount it with `<CustomShader src={MyShader}>` in any framework, no build step. Read https://shaders.com/docs/guide/custom-shaders first and the primitives reference at https://shaders.com/docs/primitives (or https://shaders.com/std/llms.txt as one document).

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
