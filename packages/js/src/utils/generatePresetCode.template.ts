interface ComponentConfig {
  type: string
  id?: string
  props?: Record<string, any>
  children?: ComponentConfig[]
}

interface PresetConfig {
  components: ComponentConfig[]
}

// Default transform values
const DEFAULT_TRANSFORM = {
  offsetX: 0,
  offsetY: 0,
  rotation: 0,
  scale: 1,
  anchorX: 0.5,
  anchorY: 0.5,
  edges: 'transparent'
}

// Bounding-box reduction helper (reduceBoundingBoxForExport) is injected here from generate-components.ts:
// __BOUNDING_BOX_EXPORT__

// __CUSTOM_COMPONENTS_EXPORT__

// @ts-ignore
const shaderMetadata: Record<string, Record<string, any>> = __SHADER_METADATA__

type PropMapValue = { type: 'map'; source: string; [key: string]: unknown }

function isPropMapValue(value: unknown): value is PropMapValue {
  return typeof value === 'object' && value !== null && 'type' in value && (value as Record<string, unknown>).type === 'map'
}

function escapeString(str: string): string {
  return str
    .replace(/\\/g, '\\\\')
    .replace(/'/g, "\\'")
    .replace(/\n/g, '\\n')
    .replace(/\r/g, '\\r')
    .replace(/\t/g, '\\t')
}

function formatValue(value: any, indent: string): string {
  if (typeof value === 'string') {
    // Detect JSON-encoded object strings and format as objects
    if (value.startsWith('{') && value.endsWith('}')) {
      try {
        const parsed = JSON.parse(value)
        if (typeof parsed === 'object' && parsed !== null) {
          return formatValue(parsed, indent)
        }
      } catch { /* not JSON, treat as string */ }
    }
    return `'${escapeString(value)}'`
  }
  if (typeof value === 'boolean' || typeof value === 'number') {
    return String(value)
  }
  if (value === null || value === undefined) {
    return String(value)
  }
  if (Array.isArray(value)) {
    const items = value.map(v => formatValue(v, indent + '  ')).join(', ')
    return `[${items}]`
  }
  if (typeof value === 'object') {
    const entries = Object.entries(value)
    if (entries.length === 0) return '{}'
    const lines = entries.map(([k, v]) => `${indent}  ${k}: ${formatValue(v, indent + '  ')}`)
    return `{\n${lines.join(',\n')}\n${indent}}`
  }
  return String(value)
}

function shouldIncludeProp(key: string, value: any, componentType: string, props: Record<string, any>): boolean {
  // Skip maskType if it's the default 'alpha'
  if (key === 'maskType' && value === 'alpha') return false

  // When a custom SDF shape URL is set, `shape` and `shapeType` are both
  // editor-only metadata — the renderer only needs `shapeSdfUrl`.
  if ((key === 'shape' || key === 'shapeType') && props.shapeSdfUrl && props.shapeType !== 'svgExtrude3D') return false

  // Special handling for transform
  if (key === 'transform' && typeof value === 'object') {
    const allDefaults = Object.keys(DEFAULT_TRANSFORM).every(
      k => value[k] === DEFAULT_TRANSFORM[k as keyof typeof DEFAULT_TRANSFORM]
    )
    if (allDefaults) return false
  }

  // Special handling for boundingBox — drop it entirely when at full-frame defaults
  if (key === 'boundingBox' && typeof value === 'object') {
    return reduceBoundingBoxForExport(value) !== null
  }

  // Group flow layout: drop when inactive; absolute: drop when falsy
  if (key === 'flow' && (!value || typeof value !== 'object' || (value as any).mode === 'none' || !(value as any).mode)) return false
  if (key === 'absolute' && !value) return false

  // Get component-specific defaults from metadata
  const componentDefaults = defaultsFor(componentType)

  // Skip if this prop matches its default value
  if (componentDefaults.hasOwnProperty(key)) {
    const defaultValue = componentDefaults[key]

    if (value != null && defaultValue != null && typeof value === 'object' && typeof defaultValue === 'object') {
      return JSON.stringify(value) !== JSON.stringify(defaultValue)
    }

    return value !== defaultValue
  }

  // Skip universal default values (fallback)
  if (key === 'opacity' && value === 1) return false
  if (key === 'blendMode' && value === 'normal') return false

  return true
}

function generateComponentObject(config: ComponentConfig, allComponents: ComponentConfig[], indent: string): string {
  const lines: string[] = []
  lines.push(`${indent}{`)

  // Type
  // A custom component is named by its definition's name; the definition is passed in options.components
  lines.push(`${indent}  type: '${customExportFor(config.type)?.name ?? config.type}',`)

  // Always include ID for JS (needed for shader.update())
  if (config.id) {
    lines.push(`${indent}  id: '${config.id}',`)
  }

  // Props
  const filteredProps: [string, any][] = []
  if (config.props) {
    for (const [key, value] of Object.entries(config.props).sort(([a], [b]) => a.localeCompare(b))) {
      if (shouldIncludeProp(key, value, config.type, config.props)) {
        // Handle transform — only include non-default keys
        if (key === 'transform' && typeof value === 'object') {
          const nonDefaultKeys: Record<string, any> = {}
          for (const k in value) {
            if (value[k] !== DEFAULT_TRANSFORM[k as keyof typeof DEFAULT_TRANSFORM]) {
              nonDefaultKeys[k] = value[k]
            }
          }
          if (Object.keys(nonDefaultKeys).length > 0) {
            filteredProps.push([key, nonDefaultKeys])
          }
        } else if (key === 'boundingBox' && typeof value === 'object') {
          // boundingBox passed shouldIncludeProp with at least one non-default key — emit only those keys
          const reduced = reduceBoundingBoxForExport(value)
          if (reduced) {
            filteredProps.push([key, reduced])
          }
        } else if (typeof value === 'object' && value !== null && 'x' in value && typeof value.x === 'number') {
          // Round x/y position values
          const rounded = { ...value }
          if (typeof rounded.x === 'number') rounded.x = Math.round(rounded.x * 100) / 100
          if (typeof rounded.y === 'number') rounded.y = Math.round(rounded.y * 100) / 100
          filteredProps.push([key, rounded])
        } else {
          filteredProps.push([key, value])
        }
      }
    }
  }

  if (filteredProps.length > 0) {
    lines.push(`${indent}  props: {`)
    for (const [key, value] of filteredProps) {
      lines.push(`${indent}    ${key}: ${formatValue(value, indent + '    ')},`)
    }
    lines.push(`${indent}  },`)
  }

  // Children
  if (config.children && config.children.length > 0) {
    lines.push(`${indent}  children: [`)
    for (const child of config.children) {
      lines.push(generateComponentObject(child, allComponents, indent + '    ') + ',')
    }
    lines.push(`${indent}  ],`)
  }

  lines.push(`${indent}}`)
  return lines.join('\n')
}

// Number-rounding helpers (roundPresetForExport) are injected here from generate-components.ts:
// __PRESET_EXPORT_ROUNDING__

export function generatePresetCode(preset: PresetConfig, colorSpace?: 'p3-linear' | 'srgb', toneMapping?: 'linear' | 'reinhard' | 'cineon' | 'aces' | 'agx' | 'neutral' | 'hable' | 'unreal', options: GeneratePresetCodeOptions = {}): string {
  customExports = options.customComponents ?? {}
  preset = roundPresetForExport(preset)
  const componentsStr = preset.components
    .map(config => generateComponentObject(config, preset.components, '    '))
    .join(',\n')

  // Custom components the preset names by type, handed to createShader so it can resolve them
  const types = new Set<string>()
  const collect = (components: ComponentConfig[]) => {
    for (const config of components) {
      types.add(config.type)
      if (config.children) collect(config.children)
    }
  }
  collect(preset.components)
  const customImports = customImportLines(types)
  const customIdentifiers = [...new Set([...types].map(type => customExportFor(type)?.identifier).filter(Boolean))].sort()

  const optionsLines: string[] = []
  if (customIdentifiers.length > 0) {
    optionsLines.push(`  components: [${customIdentifiers.join(', ')}],`)
  }
  if (colorSpace && colorSpace !== 'p3-linear') {
    optionsLines.push(`  colorSpace: '${colorSpace}',`)
  }
  if (toneMapping && toneMapping !== 'linear') {
    optionsLines.push(`  toneMapping: '${toneMapping}',`)
  }
  const optionsStr = optionsLines.length > 0
    ? `, {\n${optionsLines.join('\n')}\n}`
    : ''

  return `${["import { createShader } from 'shaders/js'", ...customImports].join('\n')}

const shader = await createShader(document.getElementById("canvas"), {
  components: [
${componentsStr}
  ]
}${optionsStr})`
}

// Available components (auto-generated)
export const availableComponents = [
  // @ts-ignore - replaced at build time
  __COMPONENT_LIST__
]
