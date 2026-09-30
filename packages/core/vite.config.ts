import {defineConfig} from 'vite'
import dts from 'vite-plugin-dts'
import typegpu from 'unplugin-typegpu/vite'
import {externalizeDeps} from 'vite-plugin-externalize-deps'
import {resolve} from 'path'
import fs from 'fs'
import * as path from "node:path";

/**
 * Get all shader entry points
 */
function getShaderEntries() {
    const shaderDir = resolve(__dirname, 'src/shaders')
    const utilsDir = resolve(__dirname, 'src/utilities')
    const shaderDirs = fs.readdirSync(shaderDir, { withFileTypes: true })
        .filter(dirent => dirent.isDirectory())
    const entries: { [key: string]: string} = {
        'index': resolve(__dirname, 'src/index.ts'),
        'registry': resolve(__dirname, 'src/registry.ts'),
        'telemetry/index': resolve(__dirname, 'src/telemetry/index.ts'),
        'utilities/transformations/index': resolve(utilsDir, 'transformations.ts'),
        // The public authoring surface (`shaders/std`): defineShader, wgsl, the vocabulary.
        'std/index': resolve(__dirname, 'src/std/index.ts'),
    }
    shaderDirs.forEach(dir => {
        const shaderPath = resolve(shaderDir, dir.name, 'index.ts')
        if (fs.existsSync(shaderPath)) {
            entries[`shaders/${dir.name}/index`] = shaderPath
        }
    })
    return entries
}

/**
 * Deprecated (former) shader names declared in a shader's `deprecatedNames: [...]` field, read by
 * TEXT SCAN of the shader source (not a module import — this runs before anything is built). The
 * field is a pure string-literal array by convention (see GpuShaderDefinition.deprecatedNames),
 * so the scan is reliable. Old names get ALIAS export entries pointing at the canonical folder's
 * dist output — old deep imports (`shaders-core/DOMTexture`) keep resolving, zero duplication.
 */
function readDeprecatedNames(shaderIndexPath: string): string[] {
    const source = fs.readFileSync(shaderIndexPath, 'utf8')
    const match = source.match(/deprecatedNames:\s*\[([^\]]*)\]/)
    if (!match) return []
    return [...match[1].matchAll(/['"`]([^'"`]+)['"`]/g)].map(m => m[1])
}

/**
 * Generate package.json exports field
 */
function generateExports() {
    const shaderDir = resolve(__dirname, 'src/shaders')
    const shaderDirs = fs.readdirSync(shaderDir, { withFileTypes: true })
        .filter(dirent => dirent.isDirectory())

    const exports: { [key: string]: { import: string; types: string; require: string } } = {
        '.': {
            types: './dist/index.d.ts',
            import: './dist/index.js',
            require: './dist/index.js',
        },
        './registry': {
            types: './dist/registry.d.ts',
            import: './dist/registry.js',
            require: './dist/registry.js',
        },
        './telemetry': {
            types: './dist/telemetry/index.d.ts',
            import: './dist/telemetry/index.js',
            require: './dist/telemetry/index.js',
        },
        './utilities/transformations': {
            types: './dist/utilities/transformations/index.d.ts',
            import: './dist/utilities/transformations/index.js',
            require: './dist/utilities/transformations/index.js',
        },
        './std': {
            types: './dist/std/index.d.ts',
            import: './dist/std/index.js',
            require: './dist/std/index.js',
        }
    }

    shaderDirs.forEach(dir => {
        const shaderPath = resolve(shaderDir, dir.name, 'index.ts')
        if (fs.existsSync(shaderPath)) {
            const entry = {
                types: `./dist/shaders/${dir.name}/index.d.ts`,
                import: `./dist/shaders/${dir.name}/index.js`,
                require: `./dist/shaders/${dir.name}/index.js`,
            }
            exports[`./${dir.name}`] = entry
            for (const oldName of readDeprecatedNames(shaderPath)) {
                exports[`./${oldName}`] = entry
            }
        }
    })

    return exports
}

/**
 * Update package.json with generated exports
 */
function updatePackageJson() {
    const packageJsonPath = resolve(__dirname, 'package.json')
    const packageJson = JSON.parse(fs.readFileSync(packageJsonPath, 'utf8'))
    packageJson.exports = generateExports()
    fs.writeFileSync(packageJsonPath, JSON.stringify(packageJson, null, 2) + '\n')
}

/**
 * Generate shader registry file
 */
function generateShaderRegistry() {
    const shaderDir = resolve(__dirname, 'src/shaders')
    const registryPath = resolve(__dirname, 'src/shaderRegistry.ts')
    
    const shaderDirs = fs.readdirSync(shaderDir, { withFileTypes: true })
        .filter(dirent => dirent.isDirectory())
        .filter(dirent => {
            const shaderPath = resolve(shaderDir, dirent.name, 'index.ts')
            return fs.existsSync(shaderPath)
        })
        .sort((a, b) => a.name.localeCompare(b.name)) // Sort for consistent output

    // Generate imports
    const imports = shaderDirs.map(dir => 
        `import { componentDefinition as ${dir.name} } from './shaders/${dir.name}/index'`
    ).join('\n')

    // Generate shader definitions object
    const shaderDefinitions = shaderDirs.map(dir => `    ${dir.name}`).join(',\n')

    const registryContent = `// Auto-generated shader registry - DO NOT EDIT MANUALLY
// This file is generated by the build process

${imports}
import type { ComponentDefinition } from './types'
import type { GpuShaderDefinition } from './gpu/contract'

export interface ShaderRegistryEntry {
  name: string
  fileName: string
  category: string
  definition: ComponentDefinition<any> | GpuShaderDefinition<any>
  propsMetadata: Record<string, {
    ui?: any
    default?: any
    description?: string
  }>
}

// Create the shader registry with all metadata
const createShaderRegistry = (): Record<string, ShaderRegistryEntry> => {
  const registry: Record<string, ShaderRegistryEntry> = {}
  
  const shaderDefinitions = {
${shaderDefinitions}
  }
  
  Object.entries(shaderDefinitions).forEach(([fileName, def]) => {
    // Extract props metadata
    const propsMetadata: Record<string, any> = {}
    if (def.props) {
      Object.entries(def.props).forEach(([propName, propConfig]) => {
        if (propConfig.ui) {
          propsMetadata[propName] = {
            ui: propConfig.ui,
            default: propConfig.default,
            description: propConfig.description
          }
        }
      })
    }
    
    registry[def.name] = {
      name: def.name,
      fileName,
      category: def.category || 'Uncategorized',
      definition: def,
      propsMetadata
    }
  })
  
  return registry
}

export const shaderRegistry = createShaderRegistry()

// Deprecated (renamed) shader names → canonical name, built from each definition's
// \`deprecatedNames\`. The registry itself is keyed by canonical names ONLY (listings/UI never
// show an old name); lookups resolve through this map so old names keep working.
export const deprecatedShaderNames: Record<string, string> = (() => {
  const map: Record<string, string> = {}
  for (const entry of Object.values(shaderRegistry)) {
    for (const oldName of (entry.definition as { deprecatedNames?: string[] }).deprecatedNames ?? []) {
      map[oldName] = entry.name
    }
  }
  return map
})()

/** Canonical shader name for \`name\` — resolves deprecated names, passes everything else through. */
export function resolveShaderName(name: string): string {
  if (shaderRegistry[name]) return name
  return deprecatedShaderNames[name] ?? name
}

// Helper functions for easy access
export function getAllShaders(): ShaderRegistryEntry[] {
  return Object.values(shaderRegistry)
}

export function getShaderByName(name: string): ShaderRegistryEntry | undefined {
  return shaderRegistry[resolveShaderName(name)]
}

export function getShadersByCategory(category: string): ShaderRegistryEntry[] {
  return Object.values(shaderRegistry).filter(shader => shader.category === category)
}

export function getShaderCategories(): string[] {
  const categories = new Set(Object.values(shaderRegistry).map(shader => shader.category))
  return Array.from(categories).sort()
}
`

    fs.writeFileSync(registryPath, registryContent)
    console.log(`✅ Generated shader registry with ${shaderDirs.length} shaders`)
}

/**
 * Get the package version - reads from shaders package (source of truth)
 * Falls back to core package for standalone builds
 */
function getPackageVersion(): string {
  const shadersPkgPath = resolve(__dirname, '../shaders/package.json')
  const corePkgPath = resolve(__dirname, 'package.json')

  // Try shaders package first (source of truth for published version)
  if (fs.existsSync(shadersPkgPath)) {
    try {
      const shadersPkg = JSON.parse(fs.readFileSync(shadersPkgPath, 'utf-8'))
      console.log(`📦 Using version ${shadersPkg.version} from shaders package`)
      return shadersPkg.version
    } catch (err) {
      console.warn('⚠️  Failed to read shaders package.json, falling back to core version')
    }
  }

  // Fallback to core package for standalone builds
  const corePkg = JSON.parse(fs.readFileSync(corePkgPath, 'utf-8'))
  console.log(`📦 Using version ${corePkg.version} from core package (fallback)`)
  return corePkg.version
}

export default defineConfig({
    define: {
        __SHADERS_VERSION__: JSON.stringify(getPackageVersion())
    },
    build: {
        minify: false,
        lib: {
            entry: getShaderEntries(),
            formats: ['es'],
            fileName: (_format, entryName) => `${entryName}.js`,
        },
        rollupOptions: {},
        rolldownOptions: {
            output: {
                exports: 'named',
                format: 'es',
                entryFileNames: (chunkInfo) => {
                    const path = chunkInfo.name.replace('src/', '');
                    return `${path}.js`;
                }
            }
        }
    },
    resolve: {
        conditions: ['import', 'module', 'node'],
        alias: {
            '@coreroot': path.resolve(__dirname, './src')
        }
    },
    plugins: [
        // TGSL transpiler: embeds the tinyest AST for 'use gpu' functions at build time
        // so consumers of the built output need no plugin. Only transforms files that
        // reference typegpu/'use gpu' (earlyPruning default).
        typegpu(),
        externalizeDeps({
            // Bundle these — colorjs.io is an internal utility; typegpu must not be a
            // bare import in the published dist, or Vite apps that exclude 'shaders'
            // from optimizeDeps hit runtime dep-discovery re-optimization loops
            // (typegpu is invisible to the dep scanner when 'shaders' is excluded).
            except: ['colorjs.io', /^typegpu(\/|$)/]
        }),
        dts({
            include: ['src'],
            beforeWriteFile: (filePath, content) => ({
                filePath: filePath.replace('/src/', '/'),
                content,
            }),
        }),
        {
            name: 'shader-build-tools',
            buildStart() {
                console.log('🔧 Updating package.json exports...')
                updatePackageJson();
                console.log('📝 Generating shader registry...')
                generateShaderRegistry();
            }
        }
    ]
})
