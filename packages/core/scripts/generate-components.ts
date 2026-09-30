#!/usr/bin/env tsx
import { resolve } from 'path'
import fs from 'fs'
import { fileURLToPath } from 'url'
import { optimizeCovers } from './optimize-covers'

const __dirname = resolve(fileURLToPath(import.meta.url), '..')

interface FrameworkConfig {
  name: string
  srcDir: string
  componentExtension: string
  includeExtensionInExports: boolean
  generateIndexBeforeUtils: boolean
}

const FRAMEWORK_CONFIGS: Record<string, FrameworkConfig> = {
  react: {
    name: 'react',
    srcDir: 'src',
    componentExtension: '.tsx',
    includeExtensionInExports: false,
    generateIndexBeforeUtils: false
  },
  vue: {
    name: 'vue',
    srcDir: 'src',
    componentExtension: '.vue',
    includeExtensionInExports: true,
    generateIndexBeforeUtils: true
  },
  svelte: {
    name: 'svelte',
    srcDir: 'src/lib',
    componentExtension: '.svelte',
    includeExtensionInExports: true,
    generateIndexBeforeUtils: false
  },
  solid: {
    name: 'solid',
    srcDir: 'src',
    componentExtension: '.tsx',
    includeExtensionInExports: false,
    generateIndexBeforeUtils: false
  },
  js: {
    name: 'js',
    srcDir: 'src',
    componentExtension: '.ts',
    includeExtensionInExports: false,
    generateIndexBeforeUtils: false
  }
}

async function loadMappableProps(
  packagesDir: string,
  shaderDirs: string[]
): Promise<Record<string, string[]>> {
  const result: Record<string, string[]> = {}
  try {
    let registryPath = resolve(packagesDir, 'core/dist/registry.js')
    if (!fs.existsSync(registryPath)) {
      registryPath = resolve(packagesDir, 'core/src/shaderRegistry.ts')
    }
    if (fs.existsSync(registryPath)) {
      const { getAllShaders } = await import(`file://${registryPath}`)
      const shaders = getAllShaders()
      for (const shader of shaders) {
        if (!shaderDirs.includes(shader.name)) continue
        const mappable: string[] = []
        const props = shader.definition?.props || {}
        for (const [propName, propConfig] of Object.entries(props)) {
          const uiType = (propConfig as any)?.ui?.type
          if (Array.isArray(uiType) && uiType.includes('map')) {
            mappable.push(propName)
          } else if (uiType === 'position') {
            mappable.push(propName)
          }
        }
        result[shader.name] = mappable
      }
    }
  } catch {
    // fallback: no mappable props for any shader
  }
  return result
}

/**
 * Deprecated (former) names per canonical shader name, from each definition's `deprecatedNames`.
 * Old names become ALIAS re-exports of the canonical component — same module, zero duplication.
 */
async function loadDeprecatedNames(
  packagesDir: string,
  shaderDirs: string[]
): Promise<Record<string, string[]>> {
  const result: Record<string, string[]> = {}
  try {
    let registryPath = resolve(packagesDir, 'core/dist/registry.js')
    if (!fs.existsSync(registryPath)) {
      registryPath = resolve(packagesDir, 'core/src/shaderRegistry.ts')
    }
    if (fs.existsSync(registryPath)) {
      const { getAllShaders } = await import(`file://${registryPath}`)
      for (const shader of getAllShaders()) {
        if (!shaderDirs.includes(shader.name)) continue
        const deprecated = (shader.definition as { deprecatedNames?: string[] })?.deprecatedNames
        if (deprecated?.length) result[shader.name] = deprecated
      }
    }
  } catch {
    // fallback: no deprecated names
  }
  return result
}

async function generateComponents(
  config: FrameworkConfig,
  packageDir: string,
  packagesDir: string,
  shaderDirs: string[]
) {
  const srcDir = resolve(packageDir, config.srcDir)
  const templatePath = resolve(srcDir, `engine/component.template${config.componentExtension}`)
  const componentsDir = resolve(srcDir, 'components')

  const template = fs.readFileSync(templatePath, 'utf-8')

  if (!fs.existsSync(componentsDir)) {
    fs.mkdirSync(componentsDir, { recursive: true })
  }

  const mappablePropsMap = await loadMappableProps(packagesDir, shaderDirs)

  // Indentation for mappable prop declarations varies by framework + position in file
  const propIndent = config.name === 'svelte' ? '        ' : '  '

  for (const shader of shaderDirs) {
    const componentName = `${shader}${config.componentExtension}`
    const mappable = mappablePropsMap[shader] || []

    // Build framework-specific type substitutions
    let omitType: string
    let mappablePropDecls: string

    if (mappable.length === 0) {
      // No mappable props — Omit<T, never> = T (no change to type)
      omitType = 'never'
      mappablePropDecls = ''
    } else {
      omitType = mappable.map(p => `'${p}'`).join(' | ')
      mappablePropDecls = mappable
        .map(p => `${propIndent}${p}?: ComponentProps['${p}'] | PropDriver;`)
        .join('\n') + '\n'
    }

    const componentContent = template
      .replace(/__SHADER_NAME__/g, shader)
      .replace(/__OMIT_TYPE__/g, omitType)
      .replace(/__MAPPABLE_PROP_DECLS__/g, mappablePropDecls)

    fs.writeFileSync(resolve(componentsDir, componentName), componentContent)
  }
}

async function generateIndex(
  config: FrameworkConfig,
  packageDir: string,
  shaderDirs: string[],
  deprecatedNamesMap: Record<string, string[]> = {}
) {
  const srcDir = resolve(packageDir, config.srcDir)

  const shaderExports = shaderDirs
    .map(shader => {
      const exportPath = config.includeExtensionInExports
        ? `./components/${shader}${config.componentExtension}`
        : `./components/${shader}`
      const lines = [`export { default as ${shader} } from '${exportPath}';`]
      // Back-compat aliases for renamed shaders: the SAME component under its old export name
      // (no duplicate module), marked @deprecated so IDEs strike it through.
      for (const oldName of deprecatedNamesMap[shader] ?? []) {
        lines.push(`/** @deprecated Renamed to \`${shader}\` — this alias will be removed in a future major version. */`)
        lines.push(`export { default as ${oldName} } from '${exportPath}';`)
      }
      return lines.join('\n')
    })
    .join('\n')

  const shaderComponentPath = config.includeExtensionInExports
    ? `./engine/Shader${config.componentExtension}`
    : `./engine/Shader`

  const previewComponentPath = config.includeExtensionInExports
    ? `./engine/Preview${config.componentExtension}`
    : `./engine/Preview`

  const shaderExport = `export { default as Shader } from '${shaderComponentPath}';`
  const previewExport = config.name === 'react'
    ? `export { Preview } from '${previewComponentPath}';`
    : `export { default as Preview } from '${previewComponentPath}';`

  // <CustomShader src={definition}> — mounts a user-defined shader (a `defineShader` result)
  // through the same registration path as the generated components.
  const customShaderComponentPath = config.includeExtensionInExports
    ? `./engine/CustomShader${config.componentExtension}`
    : `./engine/CustomShader`
  const customShaderExport = config.name === 'react'
    ? `export { CustomShader } from '${customShaderComponentPath}';\nexport type { CustomShaderProps, CustomShaderLayerProps } from '${customShaderComponentPath}';`
    : config.name === 'solid'
      ? `export { default as CustomShader } from '${customShaderComponentPath}';\nexport type { CustomShaderProps, CustomShaderLayerProps } from '${customShaderComponentPath}';`
      : `export { default as CustomShader } from '${customShaderComponentPath}';`

  // WebGPU availability helpers. Shaders is WebGPU-only, so every framework entry point
  // must let an app ask "can this browser run me?" and render its own fallback instead —
  // without reaching for `shaders/core`.
  const supportExports = [
    `export { isWebGPUSupported, getWebGPUSupport, setShadersDebug, isShadersDebug } from 'shaders-core';`,
    `export type { GpuFailureReason, WebGPUSupportInfo } from 'shaders-core';`,
    `export { registerShader, unregisterShader, getRegisteredShader, getRegisteredShaders, onShaderRegistered } from 'shaders-core';`,
    `export type { GpuShaderDefinition as ShaderDefinition } from 'shaders-core';`,
  ].join('\n')

  fs.writeFileSync(
    resolve(srcDir, 'index.ts'),
    shaderExports + '\n' + shaderExport + '\n' + previewExport + '\n' + customShaderExport + '\n' + supportExports + '\n'
  )
}

// Shared across every framework's generatePresetCode: injected into the template at the
// `// __PRESET_EXPORT_ROUNDING__` marker. Rounds all numbers in the preset before codegen so
// exported props stay readable — 0.1°-precise rotations, fit-to-frame scales, etc. don't dump
// 16-digit floats. References the PresetConfig/ComponentConfig interfaces declared in the template.
const PRESET_EXPORT_ROUNDING_SNIPPET = `// Round every number in the preset (incl. numbers inside the JSON-encoded \`shape\` string) to 4
// decimals — well below any visible threshold for shader props — so exported code stays clean.
function roundForExportDeep(value: any): any {
  if (typeof value === 'number') return Number.isFinite(value) ? Math.round(value * 1e4) / 1e4 : value
  if (typeof value === 'string') {
    if (value.startsWith('{') && value.endsWith('}')) {
      try {
        const parsed = JSON.parse(value)
        if (parsed && typeof parsed === 'object') return JSON.stringify(roundForExportDeep(parsed))
      } catch { /* not JSON — leave as-is */ }
    }
    return value
  }
  if (Array.isArray(value)) return value.map(roundForExportDeep)
  if (value && typeof value === 'object') {
    const out: Record<string, any> = {}
    for (const k in value) out[k] = roundForExportDeep(value[k])
    return out
  }
  return value
}
function roundPresetForExport(preset: PresetConfig): PresetConfig {
  const mapComp = (c: ComponentConfig): ComponentConfig => ({
    ...c,
    props: c.props ? roundForExportDeep(c.props) : c.props,
    children: c.children ? c.children.map(mapComp) : c.children
  })
  return { components: preset.components.map(mapComp) }
}`

// Shared across every framework's generatePresetCode: injected into the template at the
// `// __BOUNDING_BOX_EXPORT__` marker. Reduces a boundingBox prop to only its non-default keys
// so a box the Design Editor materialised (any touch writes all fields) doesn't bloat exports.
const BOUNDING_BOX_EXPORT_SNIPPET = `// Reduce a boundingBox prop for export: drop keys still at their full-frame defaults and strip
// editor-only fields (lockAspect — a Design Editor resize hint the renderer ignores). Returns
// null when every key is default: the box is the full-frame identity and the prop should be
// omitted entirely. The runtime wrappers merge partial boxes back with the same defaults, so
// emitting only the changed keys round-trips exactly.
function reduceBoundingBoxForExport(bb: any): Record<string, any> | null {
  if (!bb || typeof bb !== 'object') return null
  // A zero offset is zero in any unit; a full-frame extent is only knowable in UV ('%' is its UI alias)
  const isUVUnit = (u: any) => u === undefined || u === 'uv' || u === '%'
  const isZeroDim = (d: any) => !!d && typeof d === 'object' && Number(d.value) === 0
  const isFullDim = (d: any) => !!d && typeof d === 'object' && Number(d.value) === 1 && isUVUnit(d.unit)
  const out: Record<string, any> = {}
  if (bb.x !== undefined && !isZeroDim(bb.x)) out.x = bb.x
  if (bb.y !== undefined && !isZeroDim(bb.y)) out.y = bb.y
  if (bb.width !== undefined && !isFullDim(bb.width)) out.width = bb.width
  if (bb.height !== undefined && !isFullDim(bb.height)) out.height = bb.height
  if (bb.origin !== undefined && bb.origin !== 'top-left') out.origin = bb.origin
  if (bb.rotation !== undefined && Number(bb.rotation) !== 0) out.rotation = bb.rotation
  if (bb.cornerRadius !== undefined && !isZeroDim(bb.cornerRadius)) out.cornerRadius = bb.cornerRadius
  return Object.keys(out).length > 0 ? out : null
}`

async function generateUtils(
  config: FrameworkConfig,
  packageDir: string,
  packagesDir: string,
  shaderDirs: string[]
) {
  const srcDir = resolve(packageDir, config.srcDir)
  const utilsDir = resolve(srcDir, 'utils')

  if (!fs.existsSync(utilsDir)) {
    fs.mkdirSync(utilsDir, { recursive: true })
  }

  const codeGenTemplatePath = resolve(utilsDir, 'generatePresetCode.template.ts')
  if (!fs.existsSync(codeGenTemplatePath)) {
    return
  }

  const codeGenTemplate = fs.readFileSync(codeGenTemplatePath, 'utf-8')

  // Generate component list
  const componentList = shaderDirs
    .map(shader => {
      return `  '${shader}'`
    })
    .join(',\n')

  // Generate shader metadata for default value filtering
  // Import from the core package's built output to extract defaults from shader definitions
  let shaderMetadata: Record<string, Record<string, any>> = {}

  try {
    // Try built registry first, fall back to source
    let registryPath = resolve(packagesDir, 'core/dist/registry.js')

    if (!fs.existsSync(registryPath)) {
      // If dist doesn't exist, import from source
      registryPath = resolve(packagesDir, 'core/src/shaderRegistry.ts')
    }

    if (fs.existsSync(registryPath)) {
      // Import the shader registry
      const { getAllShaders } = await import(`file://${registryPath}`)
      const shaders = getAllShaders()

      // Extract default values from each shader definition
      for (const shader of shaders) {
        if (shaderDirs.includes(shader.name)) {
          const defaults: Record<string, any> = {
            opacity: 1,
            blendMode: 'normal'
          }

          // Extract from propsMetadata which has UI-visible props with defaults
          if (shader.propsMetadata) {
            for (const [propName, propMeta] of Object.entries(shader.propsMetadata)) {
              if ((propMeta as any).default !== undefined) {
                defaults[propName] = (propMeta as any).default
              }
            }
          }

          // If propsMetadata is empty, try definition.props as fallback
          if (Object.keys(defaults).length === 2 && shader.definition?.props) {
            for (const [propName, propConfig] of Object.entries(shader.definition.props)) {
              if ((propConfig as any).default !== undefined) {
                defaults[propName] = (propConfig as any).default
              }
            }
          }

          shaderMetadata[shader.name] = defaults
        }
      }
    }

    // Add fallback defaults for any shaders not found in registry
    for (const shader of shaderDirs) {
      if (!shaderMetadata[shader]) {
        shaderMetadata[shader] = {
          opacity: 1,
          blendMode: 'normal'
        }
      }
    }
  } catch (error) {
    console.warn('Could not load shader definitions from core package, using fallback defaults:', error)
    // Fallback: basic defaults for all shaders
    for (const shader of shaderDirs) {
      shaderMetadata[shader] = {
        opacity: 1,
        blendMode: 'normal'
      }
    }
  }

  const codeGenContent = codeGenTemplate
    .replace('// __PRESET_EXPORT_ROUNDING__', PRESET_EXPORT_ROUNDING_SNIPPET)
    .replace('// __BOUNDING_BOX_EXPORT__', BOUNDING_BOX_EXPORT_SNIPPET)
    .replace('__COMPONENT_LIST__', componentList)
    .replace('__SHADER_METADATA__', JSON.stringify(shaderMetadata, null, 2))

  fs.writeFileSync(resolve(utilsDir, 'generatePresetCode.ts'), codeGenContent)
}

async function generatePreviewComponentMap(
  config: FrameworkConfig,
  packageDir: string,
  shaderDirs: string[],
  deprecatedNamesMap: Record<string, string[]> = {}
) {
  const srcDir = resolve(packageDir, config.srcDir)
  const previewPath = resolve(srcDir, `engine/Preview${config.componentExtension}`)

  if (!fs.existsSync(previewPath)) return

  let content = fs.readFileSync(previewPath, 'utf-8')

  const startMarker = '// <<< SHADERS_PREVIEW_MAP:START >>>'
  const endMarker = '// <<< SHADERS_PREVIEW_MAP:END >>>'
  const startIdx = content.indexOf(startMarker)
  const endIdx = content.indexOf(endMarker)

  if (startIdx === -1 || endIdx === -1) return

  // Canonical map entries + deprecated-name aliases (old preset `type` strings resolve to the
  // SAME component — no extra import/chunk).
  const mapEntriesFor = (entry: (name: string) => string): string =>
    shaderDirs
      .flatMap(s => [entry(s), ...(deprecatedNamesMap[s] ?? []).map(old => `  ${old}: ${s},`)])
      .join('\n')

  let generated: string

  if (config.name === 'vue') {
    const imports = shaderDirs
      .map(s => `import ${s} from '../components/${s}.vue'`)
      .join('\n')
    const mapEntries = mapEntriesFor(s => `  ${s},`)
    generated = `${startMarker}
${imports}

// --- Component Map ---

const componentMap: Record<string, Component> = {
${mapEntries}
}
${endMarker}`
  } else if (config.name === 'react') {
    const mapEntries = shaderDirs
      .flatMap(s => {
        const line = (key: string) => `  ${key}: lazy(() => import('../components/${s}')),`
        return [line(s), ...(deprecatedNamesMap[s] ?? []).map(line)]
      })
      .join('\n')
    generated = `${startMarker}
const componentMap: Record<string, LazyExoticComponent<ComponentType<any>>> = {
${mapEntries}
}
${endMarker}`
  } else if (config.name === 'svelte') {
    const imports = shaderDirs
      .map(s => `import ${s} from '../components/${s}.svelte'`)
      .join('\n')
    const mapEntries = mapEntriesFor(s => `  ${s},`)
    generated = `${startMarker}
${imports}

// --- Component Map ---

const componentMap: Record<string, any> = {
${mapEntries}
}
${endMarker}`
  } else if (config.name === 'solid') {
    const importNames = shaderDirs.map(s => `  ${s},`).join('\n')
    const mapEntries = mapEntriesFor(s => `  ${s},`)
    generated = `${startMarker}
import {
${importNames}
  Shader
} from '../index'

// --- Component Mapping ---

const componentMap: Record<string, SolidComponent<any>> = {
${mapEntries}
}
${endMarker}`
  } else {
    return
  }

  const newContent = content.slice(0, startIdx) + generated + content.slice(endIdx + endMarker.length)
  fs.writeFileSync(previewPath, newContent)
}

async function main() {
  // Parse CLI arguments
  const args = process.argv.slice(2)
  const frameworkArg = args.find(arg => arg.startsWith('--framework='))
  if (!frameworkArg) {
    console.error('Error: --framework argument is required')
    console.error('Usage: tsx generate-components.ts --framework=<react|vue|svelte|solid|js>')
    process.exit(1)
  }

  const framework = frameworkArg.split('=')[1]
  const config = FRAMEWORK_CONFIGS[framework]

  if (!config) {
    console.error(`Error: Unknown framework "${framework}"`)
    console.error('Available frameworks: react, vue, svelte, solid, js')
    process.exit(1)
  }

  // Determine paths
  // This script is in packages/core/scripts/
  // __dirname resolves to packages/core/scripts
  // Target package is in packages/{framework}/
  const coreDir = resolve(__dirname, '..') // packages/core
  const packagesDir = resolve(coreDir, '..') // packages
  const packageDir = resolve(packagesDir, framework) // packages/{framework}
  const shadersDir = resolve(coreDir, 'src/shaders') // packages/core/src/shaders

  // Read shader directories
  const shaderDirs = fs.readdirSync(shadersDir, { withFileTypes: true })
    .filter(dirent => dirent.isDirectory())
    .map(dirent => dirent.name)
    .filter(name => fs.existsSync(resolve(shadersDir, name, 'index.ts')))

  const deprecatedNamesMap = await loadDeprecatedNames(packagesDir, shaderDirs)

  // JS only needs generateUtils (for generatePresetCode.ts from template)
  if (config.name === 'js') {
    await generateUtils(config, packageDir, packagesDir, shaderDirs)
  } else if (config.generateIndexBeforeUtils) {
    // Vue path
    await generateComponents(config, packageDir, packagesDir, shaderDirs)
    await generateIndex(config, packageDir, shaderDirs, deprecatedNamesMap)
    await generateUtils(config, packageDir, packagesDir, shaderDirs)
  } else {
    // React/Svelte/Solid path
    await generateComponents(config, packageDir, packagesDir, shaderDirs)
    await generateUtils(config, packageDir, packagesDir, shaderDirs)
    await generateIndex(config, packageDir, shaderDirs, deprecatedNamesMap)
  }

  if (config.name !== 'js') {
    await generatePreviewComponentMap(config, packageDir, shaderDirs, deprecatedNamesMap)
  }

  console.log(`  Generated ${shaderDirs.length} components`)

  // Optimize cover images
  console.log('\n  Optimizing cover images...')
  const optimizationStats = await optimizeCovers(shadersDir)

  if (optimizationStats.processed > 0) {
    const totalSaved = optimizationStats.originalSize - optimizationStats.optimizedSize
    const totalPercentSaved = ((totalSaved / optimizationStats.originalSize) * 100).toFixed(1)
    console.log(`  💾 Optimized ${optimizationStats.processed} images (saved ${totalPercentSaved}%)`)
  } else if (optimizationStats.skipped > 0) {
    console.log(`  ✓ All ${optimizationStats.skipped} cover images already optimized`)
  }
}

main().catch(error => {
  console.error(error)
  process.exit(1)
})
