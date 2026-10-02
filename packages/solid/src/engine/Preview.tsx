import { createMemo, createSignal, createEffect, For, type JSX, type Component as SolidComponent } from 'solid-js'
import { Dynamic } from 'solid-js/web'
// <<< SHADERS_PREVIEW_MAP:START >>>
import {
  AngularBlur,
  Arc,
  Ascii,
  Aurora,
  BarShift,
  BarnDoors,
  Beam,
  Bend,
  Blob,
  BlockDissolve,
  BlockNoise,
  BlueNoise,
  Blur,
  Boids,
  BokehBlur,
  BrickPattern,
  BrightnessContrast,
  BrushedMetal,
  Bulge,
  CRTScreen,
  CarbonFiber,
  Chalkboard,
  ChannelBlur,
  CheckerWipe,
  Checkerboard,
  Chevron,
  ChromaFlow,
  ChromaticAberration,
  Chrome,
  Circle,
  ColorWheel,
  CompressionArtifacts,
  ConcentricSpin,
  ConicGradient,
  ContourLines,
  CornerPin,
  Crescent,
  Cross,
  Crystal,
  CurlNoise,
  CursorRipples,
  CursorTrail,
  DataMosh,
  DiamondGradient,
  DiamondWipe,
  DiffuseBlur,
  DisplacementMap,
  Dither,
  DotGrid,
  DropShadow,
  Duotone,
  Ellipse,
  Emboss,
  Engraving,
  ErosionNoise,
  Exposure,
  FallingLines,
  FilmGrain,
  FilmStock,
  Flip,
  FloatingParticles,
  FlowField,
  Flower,
  FlowingGradient,
  FlutedGlass,
  Fog,
  Form3D,
  FractalNoise,
  Frost,
  GaborNoise,
  Glass,
  GlassTiles,
  Glitch,
  Glow,
  Godrays,
  Goo,
  GradientMap,
  Grayscale,
  Grid,
  GridDistortion,
  Group,
  HTMLInCanvas,
  Halftone,
  Heart,
  Heatmap,
  HexGrid,
  Hologram,
  Holographic,
  HueShift,
  ImageTexture,
  InkFlow,
  Invert,
  IrisWipe,
  Irradiance,
  IsometricCubes,
  Kaleidoscope,
  KeyFrames,
  LensDistortion,
  LensFlare,
  LightEdge,
  LightLeak,
  Line,
  LinearBlur,
  LinearGradient,
  LinearWipe,
  LiquidMetal,
  Liquify,
  MagneticFilings,
  Marble,
  MeshGradient,
  Mirror,
  MultiPointGradient,
  Nebula,
  Neon,
  NoiseDissolve,
  ObjectTracker,
  Obsidian,
  PagePeel,
  Paper,
  Parallelogram,
  ParticleField,
  ParticleFlow,
  Particles,
  PerlinNoise,
  Perspective,
  PixelSort,
  PixelThrow,
  Pixelate,
  Plasma,
  Plastic,
  PolarCoordinates,
  Polygon,
  Posterize,
  Prism,
  ProgressiveBlur,
  RadialGradient,
  RadialWipe,
  RandomBars,
  ReactionDiffusion,
  RectangularCoordinates,
  ReflectivePlane,
  Repeater,
  Ring,
  RippleWipe,
  Ripples,
  RoundedRect,
  Saturation,
  Scratches,
  Sharpness,
  Shatter,
  SimplexNoise,
  SineWave,
  SliceWipe,
  Smoke,
  SmokeFill,
  SmokeFlow,
  Solarize,
  SolidColor,
  Sparkle,
  Spherize,
  Spiral,
  Star,
  Stone,
  Strands,
  Stretch,
  Stripes,
  StudioBackground,
  SunBurst,
  Surface3D,
  Swirl,
  Teardrop,
  Text,
  ThinFilm,
  TiltShift,
  TimeTrail,
  Tint,
  Trapezoid,
  TriangularGrid,
  Tritone,
  Truchet,
  Twirl,
  VHS,
  VenetianBlinds,
  Vesica,
  Vibrance,
  VideoTexture,
  Vignette,
  Voronoi,
  Voxels,
  Water,
  Watercolor,
  WaveDistortion,
  Waveform,
  WaveletNoise,
  Weave,
  WebcamTexture,
  Wool,
  WorleyNoise,
  ZoomBlur,
  Shader
} from '../index'

// --- Component Mapping ---

const componentMap: Record<string, SolidComponent<any>> = {
  AngularBlur,
  Arc,
  Ascii,
  Aurora,
  BarShift,
  BarnDoors,
  Beam,
  Bend,
  Blob,
  BlockDissolve,
  BlockNoise,
  BlueNoise,
  Blur,
  Boids,
  BokehBlur,
  BrickPattern,
  BrightnessContrast,
  BrushedMetal,
  Bulge,
  CRTScreen,
  CarbonFiber,
  Chalkboard,
  ChannelBlur,
  CheckerWipe,
  Checkerboard,
  Chevron,
  ChromaFlow,
  ChromaticAberration,
  Chrome,
  Circle,
  ColorWheel,
  CompressionArtifacts,
  ConcentricSpin,
  ConicGradient,
  ContourLines,
  CornerPin,
  Crescent,
  Cross,
  Crystal,
  CurlNoise,
  CursorRipples,
  CursorTrail,
  DataMosh,
  DiamondGradient,
  DiamondWipe,
  DiffuseBlur,
  DisplacementMap,
  Dither,
  DotGrid,
  DropShadow,
  Duotone,
  Ellipse,
  Emboss,
  Engraving,
  ErosionNoise,
  Exposure,
  FallingLines,
  FilmGrain,
  FilmStock,
  Flip,
  FloatingParticles,
  FlowField,
  Flower,
  FlowingGradient,
  FlutedGlass,
  Fog,
  Form3D,
  FractalNoise,
  Frost,
  GaborNoise,
  Glass,
  GlassTiles,
  Glitch,
  Glow,
  Godrays,
  Goo,
  GradientMap,
  Grayscale,
  Grid,
  GridDistortion,
  Group,
  HTMLInCanvas,
  DOMTexture: HTMLInCanvas,
  Halftone,
  Heart,
  Heatmap,
  HexGrid,
  Hologram,
  Holographic,
  HueShift,
  ImageTexture,
  InkFlow,
  Invert,
  IrisWipe,
  Irradiance,
  IsometricCubes,
  Kaleidoscope,
  KeyFrames,
  LensDistortion,
  LensFlare,
  LightEdge,
  LightLeak,
  Line,
  LinearBlur,
  LinearGradient,
  LinearWipe,
  LiquidMetal,
  Liquify,
  MagneticFilings,
  Marble,
  MeshGradient,
  Mirror,
  MultiPointGradient,
  Nebula,
  Neon,
  NoiseDissolve,
  ObjectTracker,
  Obsidian,
  PagePeel,
  Paper,
  Parallelogram,
  ParticleField,
  ParticleFlow,
  Particles,
  PerlinNoise,
  Perspective,
  PixelSort,
  PixelThrow,
  Pixelate,
  Plasma,
  Plastic,
  PolarCoordinates,
  Polygon,
  Posterize,
  Prism,
  ProgressiveBlur,
  RadialGradient,
  RadialWipe,
  RandomBars,
  ReactionDiffusion,
  RectangularCoordinates,
  ReflectivePlane,
  Repeater,
  Ring,
  RippleWipe,
  Ripples,
  RoundedRect,
  Saturation,
  Scratches,
  Sharpness,
  Shatter,
  SimplexNoise,
  SineWave,
  SliceWipe,
  Smoke,
  SmokeFill,
  SmokeFlow,
  Solarize,
  SolidColor,
  Sparkle,
  Spherize,
  Spiral,
  Star,
  Stone,
  Strands,
  Stretch,
  Stripes,
  StudioBackground,
  SunBurst,
  Surface3D,
  Swirl,
  Teardrop,
  Text,
  ThinFilm,
  TiltShift,
  TimeTrail,
  Tint,
  Trapezoid,
  TriangularGrid,
  Tritone,
  Truchet,
  Twirl,
  VHS,
  VenetianBlinds,
  Vesica,
  Vibrance,
  VideoTexture,
  Vignette,
  Voronoi,
  Voxels,
  Water,
  Watercolor,
  WaveDistortion,
  Waveform,
  WaveletNoise,
  Weave,
  WebcamTexture,
  Wool,
  WorleyNoise,
  ZoomBlur,
}
// <<< SHADERS_PREVIEW_MAP:END >>>

// --- Inline Types ---

interface ShaderComponentConfig {
  type: string
  id?: string
  props?: Record<string, any>
  children?: ShaderComponentConfig[]
  blendMode?: string
  opacity?: number
  renderOrder?: number
}

interface ShaderDefinition {
  components: ShaderComponentConfig[]
  structureVersion?: number
  colorSpace?: 'p3-linear' | 'srgb'
  devicePreview?: string
}

interface KeyPropTarget {
  component_id: string
  prop: string
}

interface KeyProp {
  label: string
  type: 'color' | 'range' | 'logo' | 'image'
  category?: string
  targets: KeyPropTarget[]
}

// --- Preview Decode Logic ---
// No name mapping — preview definitions contain readable JSON.

const DEFAULT_KEY = 'shaders-preview-key'

function xorCrypt(data: Uint8Array, key: string): Uint8Array {
  const keyBytes = new TextEncoder().encode(key)
  const result = new Uint8Array(data.length)
  for (let i = 0; i < data.length; i++) {
    result[i] = data[i] ^ keyBytes[i % keyBytes.length]
  }
  return result
}

function decodePreviewDefinition(encoded: string, key: string): ShaderDefinition {
  const binaryString = atob(encoded)
  const encrypted = new Uint8Array(binaryString.length)
  for (let i = 0; i < binaryString.length; i++) {
    encrypted[i] = binaryString.charCodeAt(i)
  }
  const decrypted = xorCrypt(encrypted, key)
  const json = new TextDecoder().decode(decrypted)
  try {
    return JSON.parse(json)
  } catch {
    throw new Error('Failed to decode shader definition: invalid data or incorrect key')
  }
}

// --- Configuration apply ---
// Mirrors the key-prop identification the design editor uses when it saves a
// preset. Inlined per framework because the build copies each framework dist
// into the published npm package and won't carry a cross-package import.

const RESERVED_IDENTIFIERS = new Set([
  'break', 'case', 'catch', 'class', 'const', 'continue', 'default',
  'delete', 'do', 'else', 'export', 'extends', 'finally', 'for',
  'function', 'if', 'import', 'in', 'instanceof', 'new', 'return',
  'super', 'switch', 'this', 'throw', 'try', 'typeof', 'var', 'void',
  'while', 'with', 'yield',
])

function slugifyIdentifier(label: string, fallback: string): string {
  const cleaned = label.replace(/[^A-Za-z0-9]+/g, ' ').trim()
  if (!cleaned) return fallback
  const parts = cleaned.split(/\s+/)
  const camel = parts
    .map((part, i) => i === 0 ? part.charAt(0).toLowerCase() + part.slice(1) : part.charAt(0).toUpperCase() + part.slice(1))
    .join('')
  const safe = /^[A-Za-z_$]/.test(camel) ? camel : `_${camel}`
  return RESERVED_IDENTIFIERS.has(safe) ? `_${safe}` : safe
}

function buildKeyPropIdentifierMap(keyProps: KeyProp[] | null | undefined): Map<string, KeyProp> {
  const used = new Set<string>(['style'])
  const map = new Map<string, KeyProp>()
  if (!keyProps?.length) return map
  keyProps.forEach((kp, i) => {
    if (!kp.targets?.length) return
    const base = slugifyIdentifier(kp.label, `keyProp${i}`)
    let ident = base
    let suffix = 1
    while (used.has(ident)) {
      suffix++
      ident = `${base}${suffix}`
    }
    used.add(ident)
    map.set(ident, kp)
  })
  return map
}

function findInTree(id: string, list: ShaderComponentConfig[]): ShaderComponentConfig | null {
  for (const c of list) {
    if (c.id === id) return c
    if (c.children?.length) {
      const hit = findInTree(id, c.children)
      if (hit) return hit
    }
  }
  return null
}

function applyConfiguration(
  definition: ShaderDefinition,
  configuration: Record<string, unknown> | null | undefined,
  keyProps: KeyProp[] | null | undefined
): ShaderDefinition {
  if (!configuration || !keyProps?.length) return definition
  const entries = Object.entries(configuration)
  if (entries.length === 0) return definition
  const identMap = buildKeyPropIdentifierMap(keyProps)
  const cloned = JSON.parse(JSON.stringify(definition.components)) as ShaderComponentConfig[]
  for (const [ident, value] of entries) {
    const kp = identMap.get(ident)
    if (!kp) continue
    for (const t of kp.targets ?? []) {
      const comp = findInTree(t.component_id, cloned)
      if (comp?.props && t.prop in comp.props) comp.props[t.prop] = value
    }
  }
  return { ...definition, components: cloned }
}

// --- Recursive Renderer ---

function RenderComponent(props: {
  config: ShaderComponentConfig
  structureVersion?: number
}) {
  const Component = componentMap[props.config.type]
  if (!Component) return null

  const componentProps = props.config.props ? { ...props.config.props } : {}

  return (
    <Dynamic
      component={Component}
      id={props.config.id}
      renderOrder={props.config.renderOrder}
      {...componentProps}
    >
      <For each={props.config.children}>
        {(child, index) => (
          <RenderComponent
            config={{ ...child, renderOrder: child.renderOrder ?? index() }}
            structureVersion={props.structureVersion}
          />
        )}
      </For>
    </Dynamic>
  )
}

// --- Preview Component ---

interface StyleMap {
  [key: string]: string | number
}

function parseStyleString(style: string): JSX.CSSProperties {
  const map = style.split(';').reduce<StyleMap>((acc, declaration) => {
    const [prop, ...rest] = declaration.split(':')
    if (!prop?.trim()) return acc
    const value = rest.join(':').trim()
    const trimmed = prop.trim()
    const finalProp = trimmed.startsWith('--') ? trimmed : trimmed.replace(/-([a-z])/g, (_, c) => c.toUpperCase())
    acc[finalProp] = value
    return acc
  }, {})
  return map as JSX.CSSProperties
}

export interface PreviewProps {
  shader?: string
  presetId?: string
  apiBaseUrl?: string
  obfuscationKey?: string
  style?: JSX.CSSProperties | string
  class?: string
  watermarkText?: string
  watermarkLink?: string
  /**
   * Map of camelCase key_prop identifier → user value, layered on top of the
   * preset's authored defaults. Identifiers are derived from `key_prop.label`
   * via the same slugify rules the Pro Framer exporter uses, so identifiers
   * generated by codegen and identifiers expected here always match.
   *
   * Only applies on the `presetId` path — `shader` token previews don't carry
   * key_props and silently ignore this prop.
   */
  configuration?: Record<string, unknown>
  /**
   * Lock the preview to a specific snapshot of the preset, addressed by
   * content hash. Customized snippets pin a version so subsequent upstream
   * edits don't silently break the configuration. Server falls back to
   * latest with the `X-Shaders-Version-Fallback` header when the requested
   * snapshot is missing. Only meaningful on the `presetId` path.
   */
  version?: string
}

let hasWarnedLicense = false

export default function Preview(props: PreviewProps) {
  // Log licensing warning once
  if (!hasWarnedLicense) {
    hasWarnedLicense = true
    console.warn(
      '[Shaders] The Preview component requires a Shaders license for production use. ' +
      'Visit https://shaders.com for more information.'
    )
  }

  const [fetchedDefinition, setFetchedDefinition] = createSignal<ShaderDefinition | null>(null)
  const [fetchedKeyProps, setFetchedKeyProps] = createSignal<KeyProp[] | null>(null)

  // Helper to fetch and decode from API
  function fetchAndDecode(url: string) {
    const key = props.obfuscationKey || DEFAULT_KEY
    setFetchedDefinition(null)
    setFetchedKeyProps(null)
    fetch(url)
      .then(res => {
        if (!res.ok) throw new Error(`Failed to fetch: ${res.status}`)
        return res.json()
      })
      .then(data => {
        const item = data.preset || data.shader
        if (item?.definition && typeof item.definition === 'string') {
          setFetchedDefinition(decodePreviewDefinition(item.definition, key))
          setFetchedKeyProps(Array.isArray(item.key_props) ? item.key_props : null)
        }
      })
      .catch(err => {
        console.error('[Shaders Preview] Failed to fetch preview:', err)
      })
  }

  // Fetch shader by token
  createEffect(() => {
    const token = props.shader
    const baseUrl = props.apiBaseUrl || 'https://shaders.com'
    if (token && typeof token === 'string') {
      fetchAndDecode(`${baseUrl}/api/preview/shader/${token}`)
    } else if (!props.presetId) {
      setFetchedDefinition(null)
      setFetchedKeyProps(null)
    }
  })

  // Fetch preset by ID. Version pins the preview to a snapshot; when
  // omitted, the server returns latest.
  createEffect(() => {
    const id = props.presetId
    const version = props.version
    const baseUrl = props.apiBaseUrl || 'https://shaders.com'
    if (id) {
      const versionParam = version ? `?version=${encodeURIComponent(version)}` : ''
      fetchAndDecode(`${baseUrl}/api/preview/preset/${id}${versionParam}`)
    } else if (!props.shader) {
      setFetchedDefinition(null)
      setFetchedKeyProps(null)
    }
  })

  const definition = createMemo<ShaderDefinition | null>(() => {
    const def = fetchedDefinition()
    if (!def) return null
    return applyConfiguration(def, props.configuration, fetchedKeyProps())
  })

  const watermarkText = () => props.watermarkText || 'Unlock your Shaders Pro license'
  const watermarkLink = () => props.watermarkLink || 'https://shaders.com/dashboard?pricing=true'

  return (
    <div
      class={props.class}
      style={{
        position: 'relative',
        width: '100%',
        height: '100%',
        ...(typeof props.style === 'string' ? parseStyleString(props.style) : props.style)
      }}
    >
      {definition() && (
        <Shader
          colorSpace={definition()!.colorSpace || 'p3-linear'}
          isPreview={true}
          style={{ width: '100%', height: '100%' }}
        >
          <For each={definition()!.components}>
            {(config, index) => (
              <RenderComponent
                config={{ ...config, renderOrder: config.renderOrder ?? index() }}
                structureVersion={definition()!.structureVersion}
              />
            )}
          </For>
        </Shader>
      )}
      <a
        href={watermarkLink()}
        target="_blank"
        rel="noopener noreferrer"
        style={{
          position: 'absolute',
          bottom: '8px',
          right: '12px',
          'font-size': '11px',
          'font-family': '-apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif',
          color: 'rgba(255, 255, 255, 0.5)',
          'text-decoration': 'none',
          'pointer-events': 'auto',
          'z-index': '10',
          'text-shadow': '0 1px 3px rgba(0, 0, 0, 0.4)',
          transition: 'color 0.2s ease'
        }}
        onMouseEnter={(e) => {
          e.currentTarget.style.color = 'rgba(255, 255, 255, 0.8)'
        }}
        onMouseLeave={(e) => {
          e.currentTarget.style.color = 'rgba(255, 255, 255, 0.5)'
        }}
      >
        {watermarkText()}
      </a>
    </div>
  )
}
