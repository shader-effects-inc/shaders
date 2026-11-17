// Framework detection for `npx shaders connect`. Reads the nearest package.json
// and maps its dependencies to a meta-framework + UI library. Deliberately
// hand-rolled: the detection libraries we evaluated (@netlify/framework-info,
// @vercel/frameworks) either mislabel current versions ("Nuxt 2" for Nuxt 4)
// or pull in a large dependency tree — and package.json is all they read too.
import { existsSync } from 'node:fs'
import path from 'node:path'
import { readPackageJSON, resolvePackageJSON, type PackageJson } from 'pkg-types'
import type { ShadersFramework } from '../config'

export type Library = ShadersFramework

export const LIBRARY_LABELS: Record<Library, string> = {
  react: 'React',
  vue: 'Vue',
  svelte: 'Svelte',
  solid: 'Solid'
}

interface MetaFramework {
  dep: string
  name: string
  /** Library the meta-framework implies, or null when it's library-agnostic (Astro, Vite). */
  library: Library | null
}

// Order matters: the first match wins, so meta-frameworks come before
// generic build tools.
const META_FRAMEWORKS: MetaFramework[] = [
  { dep: 'next', name: 'Next.js', library: 'react' },
  { dep: 'nuxt', name: 'Nuxt', library: 'vue' },
  { dep: '@sveltejs/kit', name: 'SvelteKit', library: 'svelte' },
  { dep: '@solidjs/start', name: 'SolidStart', library: 'solid' },
  { dep: 'solid-start', name: 'SolidStart', library: 'solid' },
  { dep: '@tanstack/react-start', name: 'TanStack Start', library: 'react' },
  { dep: '@remix-run/react', name: 'Remix', library: 'react' },
  { dep: '@react-router/dev', name: 'React Router', library: 'react' },
  { dep: 'gatsby', name: 'Gatsby', library: 'react' },
  { dep: 'astro', name: 'Astro', library: null },
  { dep: 'vite', name: 'Vite', library: null }
]

const LIBRARY_DEPS: Array<{ dep: string, library: Library }> = [
  { dep: 'react', library: 'react' },
  { dep: 'vue', library: 'vue' },
  { dep: 'svelte', library: 'svelte' },
  { dep: 'solid-js', library: 'solid' }
]

export interface DetectedProject {
  /** Directory containing the package.json */
  dir: string
  pkg: PackageJson
  framework: string | null
  library: Library | null
  usesTypeScript: boolean
  /** Currently installed `shaders` version range, if any */
  shadersVersion: string | null
}

function allDeps(pkg: PackageJson): Record<string, string> {
  return {
    ...(pkg.dependencies ?? {}),
    ...(pkg.devDependencies ?? {}),
    ...(pkg.peerDependencies ?? {})
  }
}

export async function detectProject(cwd: string): Promise<DetectedProject | null> {
  let pkgPath: string
  try {
    pkgPath = await resolvePackageJSON(cwd)
  } catch {
    return null
  }
  const pkg = await readPackageJSON(pkgPath)
  const deps = allDeps(pkg)
  const dir = pkgPath.replace(/[\\/]package\.json$/, '')

  const meta = META_FRAMEWORKS.find(m => m.dep in deps) ?? null
  const library = meta?.library ?? LIBRARY_DEPS.find(l => l.dep in deps)?.library ?? null

  return {
    dir,
    pkg,
    framework: meta?.name ?? null,
    library,
    usesTypeScript: 'typescript' in deps,
    shadersVersion: deps.shaders ?? null
  }
}

/** "Next.js + React", "Nuxt + Vue", "Vite + Svelte", or just "React". */
export function describeDetection(framework: string | null, library: Library): string {
  const lib = LIBRARY_LABELS[library]
  if (!framework || framework === lib) return lib
  return `${framework} + ${lib}`
}

/**
 * Where installed components land, relative to the project root. Decided
 * once by `connect` and stored in shaders.config as `outDir`; `install` falls
 * back to this for configs written before the field existed.
 *
 *   Nuxt       components/shaders          (auto-import root → <ShadersName />)
 *   SvelteKit  src/lib/components/shaders  ($lib convention)
 *   Remix / React Router  app/components/shaders
 *   everything else       src/components/shaders, or components/shaders when
 *                         the project has no src/ directory (e.g. Next.js
 *                         scaffolded without one)
 */
export function defaultOutDir(dir: string, framework: string | null): string {
  if (framework === 'Nuxt') return 'components/shaders'
  if (framework === 'SvelteKit') return 'src/lib/components/shaders'
  if (framework === 'Remix' || framework === 'React Router') return 'app/components/shaders'
  if (existsSync(path.join(dir, 'src'))) return 'src/components/shaders'
  if (existsSync(path.join(dir, 'app')) && framework !== 'Next.js') return 'app/components/shaders'
  return 'components/shaders'
}
