import { lazy, Suspense, useEffect, useMemo, useRef, useState, type ComponentType, type CSSProperties, type FC, type LazyExoticComponent } from 'react'
import { Shader } from './Shader'

// --- Preset payload types (kept in sync with the design editor's export format) ---

export interface ShaderComponent {
  type: string
  id?: string
  props?: Record<string, any>
  children?: ShaderComponent[]
  blendMode?: string
  opacity?: number
  renderOrder?: number
}

export interface ShaderDefinition {
  components: ShaderComponent[]
  structureVersion?: number
  colorSpace?: 'p3-linear' | 'srgb'
  devicePreview?: string
}

export interface KeyPropTarget {
  component_id: string
  prop: string
}

export interface KeyProp {
  label: string
  type: 'color' | 'range' | 'logo' | 'image'
  category?: string
  targets: KeyPropTarget[]
}

// --- Preview Decode Logic ---
// No name mapping — preview definitions contain readable JSON.

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

function findInTree(id: string, list: ShaderComponent[]): ShaderComponent | null {
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
  const cloned = JSON.parse(JSON.stringify(definition.components)) as ShaderComponent[]
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

// --- Component mapping (lazy-loaded React components) ---

// <<< SHADERS_PREVIEW_MAP:START >>>
const componentMap: Record<string, LazyExoticComponent<ComponentType<any>>> = {
  AngularBlur: lazy(() => import('../components/AngularBlur')),
  Arc: lazy(() => import('../components/Arc')),
  Ascii: lazy(() => import('../components/Ascii')),
  Aurora: lazy(() => import('../components/Aurora')),
  BarShift: lazy(() => import('../components/BarShift')),
  BarnDoors: lazy(() => import('../components/BarnDoors')),
  Beam: lazy(() => import('../components/Beam')),
  Bend: lazy(() => import('../components/Bend')),
  Blob: lazy(() => import('../components/Blob')),
  BlockDissolve: lazy(() => import('../components/BlockDissolve')),
  BlockNoise: lazy(() => import('../components/BlockNoise')),
  BlueNoise: lazy(() => import('../components/BlueNoise')),
  Blur: lazy(() => import('../components/Blur')),
  Boids: lazy(() => import('../components/Boids')),
  BokehBlur: lazy(() => import('../components/BokehBlur')),
  BrickPattern: lazy(() => import('../components/BrickPattern')),
  BrightnessContrast: lazy(() => import('../components/BrightnessContrast')),
  BrushedMetal: lazy(() => import('../components/BrushedMetal')),
  Bulge: lazy(() => import('../components/Bulge')),
  CRTScreen: lazy(() => import('../components/CRTScreen')),
  CarbonFiber: lazy(() => import('../components/CarbonFiber')),
  Chalkboard: lazy(() => import('../components/Chalkboard')),
  ChannelBlur: lazy(() => import('../components/ChannelBlur')),
  CheckerWipe: lazy(() => import('../components/CheckerWipe')),
  Checkerboard: lazy(() => import('../components/Checkerboard')),
  Chevron: lazy(() => import('../components/Chevron')),
  ChromaFlow: lazy(() => import('../components/ChromaFlow')),
  ChromaticAberration: lazy(() => import('../components/ChromaticAberration')),
  Chrome: lazy(() => import('../components/Chrome')),
  Circle: lazy(() => import('../components/Circle')),
  ColorWheel: lazy(() => import('../components/ColorWheel')),
  CompressionArtifacts: lazy(() => import('../components/CompressionArtifacts')),
  ConcentricSpin: lazy(() => import('../components/ConcentricSpin')),
  ConicGradient: lazy(() => import('../components/ConicGradient')),
  ContourLines: lazy(() => import('../components/ContourLines')),
  CornerPin: lazy(() => import('../components/CornerPin')),
  Crescent: lazy(() => import('../components/Crescent')),
  Cross: lazy(() => import('../components/Cross')),
  Crystal: lazy(() => import('../components/Crystal')),
  CurlNoise: lazy(() => import('../components/CurlNoise')),
  CursorRipples: lazy(() => import('../components/CursorRipples')),
  CursorTrail: lazy(() => import('../components/CursorTrail')),
  DataMosh: lazy(() => import('../components/DataMosh')),
  DiamondGradient: lazy(() => import('../components/DiamondGradient')),
  DiamondWipe: lazy(() => import('../components/DiamondWipe')),
  DiffuseBlur: lazy(() => import('../components/DiffuseBlur')),
  DisplacementMap: lazy(() => import('../components/DisplacementMap')),
  Dither: lazy(() => import('../components/Dither')),
  DotGrid: lazy(() => import('../components/DotGrid')),
  DropShadow: lazy(() => import('../components/DropShadow')),
  Duotone: lazy(() => import('../components/Duotone')),
  Ellipse: lazy(() => import('../components/Ellipse')),
  Emboss: lazy(() => import('../components/Emboss')),
  Engraving: lazy(() => import('../components/Engraving')),
  ErosionNoise: lazy(() => import('../components/ErosionNoise')),
  Exposure: lazy(() => import('../components/Exposure')),
  FallingLines: lazy(() => import('../components/FallingLines')),
  FilmGrain: lazy(() => import('../components/FilmGrain')),
  FilmStock: lazy(() => import('../components/FilmStock')),
  Flip: lazy(() => import('../components/Flip')),
  FloatingParticles: lazy(() => import('../components/FloatingParticles')),
  FlowField: lazy(() => import('../components/FlowField')),
  Flower: lazy(() => import('../components/Flower')),
  FlowingGradient: lazy(() => import('../components/FlowingGradient')),
  FlutedGlass: lazy(() => import('../components/FlutedGlass')),
  Fog: lazy(() => import('../components/Fog')),
  Form3D: lazy(() => import('../components/Form3D')),
  FractalNoise: lazy(() => import('../components/FractalNoise')),
  Frost: lazy(() => import('../components/Frost')),
  GaborNoise: lazy(() => import('../components/GaborNoise')),
  Glass: lazy(() => import('../components/Glass')),
  GlassTiles: lazy(() => import('../components/GlassTiles')),
  Glitch: lazy(() => import('../components/Glitch')),
  Glow: lazy(() => import('../components/Glow')),
  Godrays: lazy(() => import('../components/Godrays')),
  Goo: lazy(() => import('../components/Goo')),
  GradientMap: lazy(() => import('../components/GradientMap')),
  Grayscale: lazy(() => import('../components/Grayscale')),
  Grid: lazy(() => import('../components/Grid')),
  GridDistortion: lazy(() => import('../components/GridDistortion')),
  Group: lazy(() => import('../components/Group')),
  HTMLInCanvas: lazy(() => import('../components/HTMLInCanvas')),
  DOMTexture: lazy(() => import('../components/HTMLInCanvas')),
  Halftone: lazy(() => import('../components/Halftone')),
  Heart: lazy(() => import('../components/Heart')),
  Heatmap: lazy(() => import('../components/Heatmap')),
  HexGrid: lazy(() => import('../components/HexGrid')),
  Hologram: lazy(() => import('../components/Hologram')),
  Holographic: lazy(() => import('../components/Holographic')),
  HueShift: lazy(() => import('../components/HueShift')),
  ImageTexture: lazy(() => import('../components/ImageTexture')),
  InkFlow: lazy(() => import('../components/InkFlow')),
  Invert: lazy(() => import('../components/Invert')),
  IrisWipe: lazy(() => import('../components/IrisWipe')),
  Irradiance: lazy(() => import('../components/Irradiance')),
  IsometricCubes: lazy(() => import('../components/IsometricCubes')),
  Kaleidoscope: lazy(() => import('../components/Kaleidoscope')),
  KeyFrames: lazy(() => import('../components/KeyFrames')),
  LensDistortion: lazy(() => import('../components/LensDistortion')),
  LensFlare: lazy(() => import('../components/LensFlare')),
  LightEdge: lazy(() => import('../components/LightEdge')),
  LightLeak: lazy(() => import('../components/LightLeak')),
  Line: lazy(() => import('../components/Line')),
  LinearBlur: lazy(() => import('../components/LinearBlur')),
  LinearGradient: lazy(() => import('../components/LinearGradient')),
  LinearWipe: lazy(() => import('../components/LinearWipe')),
  LiquidMetal: lazy(() => import('../components/LiquidMetal')),
  Liquify: lazy(() => import('../components/Liquify')),
  MagneticFilings: lazy(() => import('../components/MagneticFilings')),
  Marble: lazy(() => import('../components/Marble')),
  MeshGradient: lazy(() => import('../components/MeshGradient')),
  Mirror: lazy(() => import('../components/Mirror')),
  MultiPointGradient: lazy(() => import('../components/MultiPointGradient')),
  Nebula: lazy(() => import('../components/Nebula')),
  Neon: lazy(() => import('../components/Neon')),
  NoiseDissolve: lazy(() => import('../components/NoiseDissolve')),
  ObjectTracker: lazy(() => import('../components/ObjectTracker')),
  Obsidian: lazy(() => import('../components/Obsidian')),
  PagePeel: lazy(() => import('../components/PagePeel')),
  Paper: lazy(() => import('../components/Paper')),
  Parallelogram: lazy(() => import('../components/Parallelogram')),
  ParticleField: lazy(() => import('../components/ParticleField')),
  ParticleFlow: lazy(() => import('../components/ParticleFlow')),
  Particles: lazy(() => import('../components/Particles')),
  PerlinNoise: lazy(() => import('../components/PerlinNoise')),
  Perspective: lazy(() => import('../components/Perspective')),
  PixelSort: lazy(() => import('../components/PixelSort')),
  PixelThrow: lazy(() => import('../components/PixelThrow')),
  Pixelate: lazy(() => import('../components/Pixelate')),
  Plasma: lazy(() => import('../components/Plasma')),
  Plastic: lazy(() => import('../components/Plastic')),
  PolarCoordinates: lazy(() => import('../components/PolarCoordinates')),
  Polygon: lazy(() => import('../components/Polygon')),
  Posterize: lazy(() => import('../components/Posterize')),
  Prism: lazy(() => import('../components/Prism')),
  ProgressiveBlur: lazy(() => import('../components/ProgressiveBlur')),
  RadialGradient: lazy(() => import('../components/RadialGradient')),
  RadialWipe: lazy(() => import('../components/RadialWipe')),
  RandomBars: lazy(() => import('../components/RandomBars')),
  ReactionDiffusion: lazy(() => import('../components/ReactionDiffusion')),
  RectangularCoordinates: lazy(() => import('../components/RectangularCoordinates')),
  ReflectivePlane: lazy(() => import('../components/ReflectivePlane')),
  Repeater: lazy(() => import('../components/Repeater')),
  Ring: lazy(() => import('../components/Ring')),
  RippleWipe: lazy(() => import('../components/RippleWipe')),
  Ripples: lazy(() => import('../components/Ripples')),
  RoundedRect: lazy(() => import('../components/RoundedRect')),
  Saturation: lazy(() => import('../components/Saturation')),
  Scratches: lazy(() => import('../components/Scratches')),
  Sharpness: lazy(() => import('../components/Sharpness')),
  Shatter: lazy(() => import('../components/Shatter')),
  SimplexNoise: lazy(() => import('../components/SimplexNoise')),
  SineWave: lazy(() => import('../components/SineWave')),
  SliceWipe: lazy(() => import('../components/SliceWipe')),
  Smoke: lazy(() => import('../components/Smoke')),
  SmokeFill: lazy(() => import('../components/SmokeFill')),
  SmokeFlow: lazy(() => import('../components/SmokeFlow')),
  Solarize: lazy(() => import('../components/Solarize')),
  SolidColor: lazy(() => import('../components/SolidColor')),
  Sparkle: lazy(() => import('../components/Sparkle')),
  Spherize: lazy(() => import('../components/Spherize')),
  Spiral: lazy(() => import('../components/Spiral')),
  Star: lazy(() => import('../components/Star')),
  Stone: lazy(() => import('../components/Stone')),
  Strands: lazy(() => import('../components/Strands')),
  Stretch: lazy(() => import('../components/Stretch')),
  Stripes: lazy(() => import('../components/Stripes')),
  StudioBackground: lazy(() => import('../components/StudioBackground')),
  SunBurst: lazy(() => import('../components/SunBurst')),
  Surface3D: lazy(() => import('../components/Surface3D')),
  Swirl: lazy(() => import('../components/Swirl')),
  Teardrop: lazy(() => import('../components/Teardrop')),
  Text: lazy(() => import('../components/Text')),
  ThinFilm: lazy(() => import('../components/ThinFilm')),
  TiltShift: lazy(() => import('../components/TiltShift')),
  TimeTrail: lazy(() => import('../components/TimeTrail')),
  Tint: lazy(() => import('../components/Tint')),
  Trapezoid: lazy(() => import('../components/Trapezoid')),
  TriangularGrid: lazy(() => import('../components/TriangularGrid')),
  Tritone: lazy(() => import('../components/Tritone')),
  Truchet: lazy(() => import('../components/Truchet')),
  Twirl: lazy(() => import('../components/Twirl')),
  VHS: lazy(() => import('../components/VHS')),
  VenetianBlinds: lazy(() => import('../components/VenetianBlinds')),
  Vesica: lazy(() => import('../components/Vesica')),
  Vibrance: lazy(() => import('../components/Vibrance')),
  VideoTexture: lazy(() => import('../components/VideoTexture')),
  Vignette: lazy(() => import('../components/Vignette')),
  Voronoi: lazy(() => import('../components/Voronoi')),
  Voxels: lazy(() => import('../components/Voxels')),
  Water: lazy(() => import('../components/Water')),
  Watercolor: lazy(() => import('../components/Watercolor')),
  WaveDistortion: lazy(() => import('../components/WaveDistortion')),
  Waveform: lazy(() => import('../components/Waveform')),
  WaveletNoise: lazy(() => import('../components/WaveletNoise')),
  Weave: lazy(() => import('../components/Weave')),
  WebcamTexture: lazy(() => import('../components/WebcamTexture')),
  Wool: lazy(() => import('../components/Wool')),
  WorleyNoise: lazy(() => import('../components/WorleyNoise')),
  ZoomBlur: lazy(() => import('../components/ZoomBlur')),
}
// <<< SHADERS_PREVIEW_MAP:END >>>

// --- Preview component ---

const DEFAULT_KEY = 'shaders-preview-key'

interface StyleMap {
  [key: string]: string | number
}

function parseStyleString(style: string): CSSProperties {
  const map = style.split(';').reduce<StyleMap>((acc, declaration) => {
    const [prop, ...rest] = declaration.split(':')
    if (!prop?.trim()) return acc
    const value = rest.join(':').trim()
    const trimmed = prop.trim()
    const finalProp = trimmed.startsWith('--') ? trimmed : trimmed.replace(/-([a-z])/g, (_, c) => c.toUpperCase())
    acc[finalProp] = value
    return acc
  }, {})
  return map as CSSProperties
}

export interface PreviewProps {
  shader?: string
  presetId?: string
  apiBaseUrl?: string
  obfuscationKey?: string
  watermarkText?: string
  watermarkLink?: string
  style?: CSSProperties | string
  className?: string
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
   * edits (label renames, key_prop regenerations) don't silently break the
   * configuration. Server falls back to latest with the
   * `X-Shaders-Version-Fallback` header when the requested snapshot is
   * missing. Only meaningful on the `presetId` path.
   */
  version?: string
  [key: string]: any
}

function RenderComponent({ component }: { component: ShaderComponent }) {
  const Component = componentMap[component.type]
  if (!Component) {
    console.warn(`[Shaders Preview] Unknown component type: ${component.type}`)
    return null
  }

  const { type, children, id, props: componentProps, ...topLevelProps } = component

  const mergedProps: Record<string, any> = {
    ...topLevelProps,
    ...componentProps,
  }

  if (id) {
    mergedProps.id = id
  }

  return (
    <Component {...mergedProps}>
      {children?.map((child, index) => (
        <RenderComponent key={child.id || index} component={child} />
      ))}
    </Component>
  )
}

export const Preview: FC<PreviewProps> = ({
  shader,
  presetId,
  apiBaseUrl = 'https://shaders.com',
  obfuscationKey = DEFAULT_KEY,
  watermarkText = 'Unlock your Shaders Pro license',
  watermarkLink = 'https://shaders.com/dashboard?pricing=true',
  style,
  className,
  configuration,
  version,
  ...rest
}) => {
  const [fetchedDefinition, setFetchedDefinition] = useState<ShaderDefinition | null>(null)
  const [fetchedKeyProps, setFetchedKeyProps] = useState<KeyProp[] | null>(null)

  // Helper to fetch and decode from API
  const fetchAndDecode = (url: string) => {
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
          setFetchedDefinition(decodePreviewDefinition(item.definition, obfuscationKey))
          setFetchedKeyProps(Array.isArray(item.key_props) ? item.key_props : null)
        }
      })
      .catch(err => {
        console.error('[Shaders Preview] Failed to fetch preview:', err)
      })
  }

  // Fetch shader by token (shader prop)
  useEffect(() => {
    if (shader) {
      fetchAndDecode(`${apiBaseUrl}/api/preview/shader/${shader}`)
    } else if (!presetId) {
      setFetchedDefinition(null)
      setFetchedKeyProps(null)
    }
  }, [shader, presetId, apiBaseUrl, obfuscationKey])

  // Fetch preset by ID. Version pins the preview to a snapshot; when omitted,
  // the server returns latest.
  useEffect(() => {
    if (presetId) {
      const versionParam = version ? `?version=${encodeURIComponent(version)}` : ''
      fetchAndDecode(`${apiBaseUrl}/api/preview/preset/${presetId}${versionParam}`)
    } else if (!shader) {
      setFetchedDefinition(null)
      setFetchedKeyProps(null)
    }
  }, [presetId, shader, apiBaseUrl, obfuscationKey, version])

  const definition = useMemo(() => {
    if (!fetchedDefinition) return null
    return applyConfiguration(fetchedDefinition, configuration, fetchedKeyProps)
  }, [fetchedDefinition, fetchedKeyProps, configuration])

  const hasWarnedRef = useRef(false)
  useEffect(() => {
    if (!hasWarnedRef.current && definition) {
      hasWarnedRef.current = true
      console.warn(
        '[Shaders] The Preview component is intended for use with a valid Shaders license. ' +
        'Please visit https://shaders.com for more information.'
      )
    }
  }, [definition])

  if (!definition) return null

  return (
    <div
      style={{ position: 'relative', ...(typeof style === 'string' ? parseStyleString(style) : style) }}
      className={className}
      {...rest}
    >
      <Suspense fallback={null}>
        <Shader
          colorSpace={definition.colorSpace || 'p3-linear'}
          isPreview={true}
          style={{ width: '100%', height: '100%' }}
        >
          {definition.components.map((component, index) => (
            <RenderComponent key={component.id || index} component={component} />
          ))}
        </Shader>
      </Suspense>

      {watermarkText && (
        <div
          style={{
            position: 'absolute',
            bottom: '8px',
            right: '12px',
            fontSize: '11px',
            fontFamily: '-apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif',
            color: 'rgba(255, 255, 255, 0.5)',
            textShadow: '0 1px 3px rgba(0, 0, 0, 0.4)',
            pointerEvents: watermarkLink ? 'auto' : 'none',
            zIndex: 10,
            transition: 'color 0.2s ease',
          }}
          onMouseEnter={(e) => {
            if (watermarkLink) {
              e.currentTarget.style.color = 'rgba(255, 255, 255, 0.8)'
            }
          }}
          onMouseLeave={(e) => {
            if (watermarkLink) {
              e.currentTarget.style.color = 'rgba(255, 255, 255, 0.5)'
            }
          }}
        >
          {watermarkLink ? (
            <a
              href={watermarkLink}
              target="_blank"
              rel="noopener noreferrer"
              style={{ color: 'inherit', textDecoration: 'none' }}
            >
              {watermarkText}
            </a>
          ) : (
            watermarkText
          )}
        </div>
      )}
    </div>
  )
}

export default Preview
