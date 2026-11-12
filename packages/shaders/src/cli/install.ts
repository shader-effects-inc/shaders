// `npx shaders install [<ref>...]` — fetch shader component source from
// shaders.com and write it into the project's outDir.
//
//   <numeric id>   one of your own saved shaders
//   <slug> / uuid  a curated preset (cloned into your project first, or the
//                  watermarked preview when you don't have Pro — see presets.ts)
//   (nothing)      pick from the shaders in the connected project
import { lstatSync, realpathSync } from 'node:fs'
import { mkdir, readFile, rm, writeFile } from 'node:fs/promises'
import path from 'node:path'
import { consola } from 'consola'
import { api, ApiError, GoneError, projectGone } from './api'
import { ensureSignedIn, getSignedInState, type Credentials } from './auth'
import { setConfigField } from './configFile'
import { cancelled, chooseProject, init, isInteractive, type InitFlags } from './init'
import { hashContent, readLockFile, writeLockFile, type ComponentLockEntry, type LockEntry, type LockFile, type LockSource } from './lockFile'
import { clonePreset, ensureProject, isPresetRef, offerPro, presetSource, renderPreviewFile, resolvePreset, type PresetSummary } from './presets'
import { hasProjectConfig, resolveProjectContext, type ProjectContext } from './project'
import type { ShadersFramework } from '../config'

export interface InstallFlags extends InitFlags {
  force: boolean
  /** Install every shader in the project without prompting */
  all: boolean
}

interface ProjectShader {
  id: string
  title: string
  updated_at: string
}

interface ProjectShadersPage {
  project: { id: string, title: string }
  shaders: ProjectShader[]
  has_more: boolean
  next_offset: number | null
}

const ALL = '__all__'
const CANCEL = Symbol.for('cancel')
const PAGE_SIZE = 100
const MAX_SHADERS = 1000

/** A file the shader component imports: a custom component's defineShader definition. */
export interface ExportFile {
  /** Relative to the shader component's folder, e.g. `custom/Halo.ts` */
  path: string
  /** The custom component's layer type, also on the file's `// shaders:type` line */
  type: string
  contents: string
}

export interface CodeResponse {
  id: string
  title: string
  format: ShadersFramework
  code: string
  /** Custom component files the code imports (absent from older servers) */
  files?: ExportFile[]
  import: string
  kind: 'shader' | 'preset'
  project_id?: string | null
  collection?: string | null
  updated_at?: string | null
}

const EXTENSIONS: Record<ShadersFramework, string> = {
  react: '.tsx',
  solid: '.tsx',
  vue: '.vue',
  svelte: '.svelte'
}

/** "Aurora Hero (v2)" → "AuroraHeroV2"; guaranteed to be a valid identifier. */
export function componentNameFromTitle(title: string, id: string): string {
  const words = title.normalize('NFKD').replace(/[^\w\s-]/g, ' ').split(/[\s_-]+/).filter(Boolean)
  let name = words.map(w => w[0].toUpperCase() + w.slice(1)).join('')
  if (!name) name = `Shader${id.replace(/[^a-zA-Z0-9]/g, '').slice(0, 8)}`
  if (/^\d/.test(name)) name = `Shader${name}`
  return name
}

/** The generators hard-code `ShaderEffect` for React/Solid; SFCs are named by file. */
function applyComponentName(code: string, name: string): string {
  return code.replace(/\bShaderEffect\b/g, name)
}

export function usageHint(ctx: ProjectContext, entry: LockEntry): string {
  const withoutExt = entry.file.replace(/\.[^.]+$/, '')
  if (ctx.detected.framework === 'Nuxt') {
    // Nuxt auto-imports with the directory path as a prefix
    const rel = path.posix.relative('components', path.posix.dirname(entry.file))
    const prefix = rel && !rel.startsWith('..')
      ? rel.split('/').map(s => s ? s[0].toUpperCase() + s.slice(1) : '').join('')
      : ''
    return `<${prefix}${entry.component} />  (auto-imported)`
  }
  if (ctx.detected.framework === 'SvelteKit' && entry.file.startsWith('src/lib/')) {
    return `import ${entry.component} from '$lib/${entry.file.slice('src/lib/'.length)}'`
  }
  const spec = ctx.framework === 'vue' || ctx.framework === 'svelte' ? entry.file : withoutExt
  return `import ${entry.component} from './${spec}'  (relative to the project root)`
}

export async function fetchCode(credentials: Credentials, id: string, framework: ShadersFramework): Promise<CodeResponse> {
  try {
    return await api<CodeResponse>(`/api/plugin/shaders/${encodeURIComponent(id)}/code`, {
      token: credentials.accessToken,
      query: { format: framework }
    })
  } catch (error) {
    if (error instanceof ApiError && error.status === 403 && error.upgradeUrl) {
      throw new Error(`${error.message}\n  ${error.upgradeUrl}`)
    }
    if (error instanceof ApiError && error.isProjectNotFound) throw projectGone(id)
    if (error instanceof ApiError && error.status === 410 && error.code === 'shader_deleted') {
      const title = typeof error.data?.title === 'string' ? error.data.title : `Shader ${id}`
      throw new GoneError(`"${title}" has been deleted from Shaders. Remove it from shaders.lock.json to stop tracking it.`, 'shader')
    }
    if (error instanceof ApiError && error.status === 404) {
      throw new GoneError(`Shader ${id} has been deleted from Shaders (or isn't in your account). IDs are shown in the design editor's Export Code dialog.`, 'shader')
    }
    throw error
  }
}

// ---------------------------------------------------------------------------
// Destination safety
// ---------------------------------------------------------------------------

/**
 * Paths come from shaders.lock.json / shaders.config, which may have been
 * committed by someone else — never let them escape the project root.
 * (The root, not outDir: a user who changes outDir still owns files
 * installed under the old one, and update must keep reaching them.)
 */
export function resolveInsideProject(projectDir: string, relFile: string): string {
  const root = path.resolve(projectDir)
  const abs = path.resolve(root, relFile)
  const inside = (candidate: string, base: string) => candidate === base || candidate.startsWith(base + path.sep)

  // Lexical check first (catches `..` and absolute paths)…
  if (!inside(abs, root)) {
    throw new Error(`Refusing to write outside the project: ${relFile}`)
  }
  // …then follow symlinks: a committed `src/components/shaders -> /elsewhere`
  // link would pass the lexical check while the bytes land outside the
  // project. Resolve the deepest ancestor that exists on disk (the file
  // itself may not yet) and compare real paths.
  // lstat (not exists/stat) so a dangling symlink counts as present — writing
  // through one would create its target, wherever that points.
  const existsNoFollow = (p: string) => { try { lstatSync(p); return true } catch { return false } }
  let probe = abs
  while (!existsNoFollow(probe)) probe = path.dirname(probe)
  let realProbe: string
  try {
    realProbe = realpathSync(probe)
  } catch {
    throw new Error(`Refusing to write through a broken symlink: ${relFile}`)
  }
  if (!inside(realProbe, realpathSync(root))) {
    throw new Error(`Refusing to write outside the project (symlink escapes it): ${relFile}`)
  }
  return abs
}

/** File contents, or null only when it doesn't exist. Other errors propagate. */
export async function readIfExists(file: string): Promise<string | null> {
  try {
    return await readFile(file, 'utf8')
  } catch (error) {
    if ((error as NodeJS.ErrnoException).code === 'ENOENT') return null
    throw error
  }
}

// ---------------------------------------------------------------------------
// Writing files
// ---------------------------------------------------------------------------

/**
 * Component name + file for a fresh install. Two different shaders can share
 * a title ("Hero" in two projects), so a name already claimed by another lock
 * entry gets the shader's id appended rather than overwriting that file.
 */
function pickFreeName(ctx: ProjectContext, title: string, id: string, lock: LockFile): { component: string, relFile: string } {
  const outDir = ctx.outDir.split(path.sep).join('/')
  const ext = EXTENSIONS[ctx.framework]
  const taken = new Set(
    Object.entries(lock.shaders).filter(([key]) => key !== id).map(([, e]) => e.file)
  )
  const base = componentNameFromTitle(title, id)
  const idSuffix = id.replace(/[^a-zA-Z0-9]/g, '').slice(0, 8)
  const candidates = [base, `${base}${idSuffix}`]
  for (let n = 2; ; n++) {
    for (const component of candidates) {
      const relFile = path.posix.join(outDir, `${component}${ext}`)
      if (!taken.has(relFile)) return { component, relFile }
    }
    candidates.splice(0, candidates.length, `${base}${idSuffix}${n}`)
  }
}

export interface WriteResult {
  entry: LockEntry
  status: 'installed' | 'updated' | 'unchanged'
  /** Custom component files written or removed alongside the shader */
  components: Array<{ file: string, status: 'installed' | 'updated' | 'removed' | 'kept' }>
}

interface WriteInput {
  /** Lock key: saved-shader id, or preset id for previews */
  id: string
  title: string
  code: string
  /** Custom component files the code imports */
  files?: ExportFile[]
  updatedAt: string | null
  source?: LockSource
}

// ---------------------------------------------------------------------------
// Custom component files
// ---------------------------------------------------------------------------

const TYPE_LINE = /^\/\/ shaders:type (\S+)$/m

/** The `// shaders:type` a file declares in its header, if any. */
function declaredType(contents: string): string | null {
  return TYPE_LINE.exec(contents.split('\n').slice(0, 5).join('\n'))?.[1] ?? null
}

/** `custom/Halo.ts` → `custom/Halo-2.ts` */
function withSuffix(relFile: string, n: number): string {
  const ext = path.posix.extname(relFile)
  return n === 1 ? relFile : `${relFile.slice(0, -ext.length)}-${n}${ext}`
}

/** The import specifier for `relFile` from a file in `fromDir` (both project-relative). */
function importPath(fromDir: string, relFile: string): string {
  const rel = path.posix.relative(fromDir, relFile).replace(/\.ts$/, '')
  return rel.startsWith('.') ? rel : `./${rel}`
}

/**
 * Write the custom component files a shader imports, and keep the lock's
 * `components` in step. Returns the shader code with any import paths that had
 * to change rewritten.
 *
 * - A type already installed (by any shader) keeps its file; identical
 *   components are shared.
 * - A new revision of a component this shader used takes over the old file
 *   when no other shader still imports the old one, so git shows an edit.
 * - When the suggested name is taken by a different component, the file gets
 *   a suffix (`Halo-2.ts`) and the shader's import follows it.
 * - Files this shader no longer imports are removed once nothing uses them,
 *   unless they have local edits.
 */
async function writeCustomFiles(ctx: ProjectContext, shaderKey: string, shaderFile: string, code: string, files: ExportFile[], lock: LockFile, force: boolean): Promise<{ code: string, components: WriteResult['components'] }> {
  const shaderDir = path.posix.dirname(shaderFile)
  const components: WriteResult['components'] = []
  const wanted = new Set(files.map(file => file.type))
  const previous = Object.entries(lock.components).filter(([, entry]) => entry.usedBy.includes(shaderKey))
  const claimed = new Set(Object.values(lock.components).map(entry => entry.file))
  let rewritten = code

  for (const file of files) {
    const contents = file.contents.replace(/\n?$/, '\n')
    const hash = hashContent(contents)
    let entry: ComponentLockEntry | undefined = lock.components[file.type]

    if (!entry) {
      // A revision of a component only this shader used: take over its file
      const slot = previous.find(([type, old]) => !wanted.has(type) && old.suggested === file.path && old.usedBy.every(key => key === shaderKey))
      if (slot) {
        entry = { ...slot[1], usedBy: [] }
        delete lock.components[slot[0]]
      } else {
        entry = { file: await freeComponentFile(ctx, path.posix.join(shaderDir, file.path), file.type, claimed), suggested: file.path, hash: '', usedBy: [] }
      }
      claimed.add(entry.file)
    }

    const absFile = resolveInsideProject(ctx.detected.dir, entry.file)
    const current = await readIfExists(absFile)
    if (current === null || hashContent(current) !== hash) {
      const pristine = current === null || hashContent(current) === entry.hash
      if (!pristine && !force) throw new Error(`${entry.file} has local edits — re-run with --force to overwrite them`)
      await mkdir(path.dirname(absFile), { recursive: true })
      await writeFile(absFile, contents)
      components.push({ file: entry.file, status: current === null ? 'installed' : 'updated' })
    }
    lock.components[file.type] = { ...entry, hash, usedBy: [...new Set([...entry.usedBy, shaderKey])].sort() }

    const expected = importPath(shaderDir, path.posix.join(shaderDir, file.path))
    const actual = importPath(shaderDir, entry.file)
    if (actual !== expected) rewritten = rewritten.split(`'${expected}'`).join(`'${actual}'`)
  }

  // Components this shader no longer imports
  for (const [type, entry] of previous) {
    if (wanted.has(type) || !lock.components[type]) continue
    const usedBy = entry.usedBy.filter(key => key !== shaderKey)
    if (usedBy.length) {
      lock.components[type] = { ...entry, usedBy }
      continue
    }
    delete lock.components[type]
    const absFile = resolveInsideProject(ctx.detected.dir, entry.file)
    const current = await readIfExists(absFile)
    if (current === null) continue
    if (hashContent(current) === entry.hash) {
      await rm(absFile)
      components.push({ file: entry.file, status: 'removed' })
    } else {
      components.push({ file: entry.file, status: 'kept' })
    }
  }

  return { code: rewritten, components }
}

/**
 * A free project-relative path for a new component file, starting from the
 * suggested one. A path is free when no lock entry claims it and either
 * nothing is on disk or the file there declares the same type (installed by
 * an agent over MCP, say).
 */
async function freeComponentFile(ctx: ProjectContext, suggested: string, type: string, claimed: Set<string>): Promise<string> {
  for (let n = 1; ; n++) {
    const candidate = withSuffix(suggested, n)
    if (claimed.has(candidate)) continue
    const current = await readIfExists(resolveInsideProject(ctx.detected.dir, candidate))
    if (current === null || declaredType(current) === type) return candidate
  }
}

/**
 * Write (or rewrite) one component file and return its lock entry. Shared by
 * install and update: `existing` is the current lock entry when refreshing,
 * used to keep the file name stable across title changes.
 */
export async function writeComponentFile(ctx: ProjectContext, input: WriteInput, existing: LockEntry | null, lock: LockFile, force: boolean): Promise<WriteResult> {
  const { component, relFile } = existing
    ? { component: existing.component, relFile: existing.file }
    : pickFreeName(ctx, input.title, input.id, lock)
  const absFile = resolveInsideProject(ctx.detected.dir, relFile)
  const current = await readIfExists(absFile)

  // Refuse before writing anything else when the shader file itself is protected
  const pristine = current === null || (existing ? hashContent(current) === existing.hash : false)
  const guard = () => {
    throw new Error(
      existing
        ? `${relFile} has local edits — re-run with --force to overwrite them`
        : `${relFile} already exists — re-run with --force to overwrite it`
    )
  }

  const named = applyComponentName(input.code, component)
  if (!pristine && !force && hashContent(current!) !== hashContent(named.replace(/\n?$/, '\n'))) guard()

  // Custom components first: the shader imports them, and their final paths can change its imports
  const custom = await writeCustomFiles(ctx, input.id, relFile, named, input.files ?? [], lock, force)
  const code = custom.code.replace(/\n?$/, '\n')
  const hash = hashContent(code)

  if (current !== null) {
    if (hashContent(current) === hash) {
      return { entry: makeEntry(existing, component, relFile, ctx.framework, hash, input), status: 'unchanged', components: custom.components }
    }
    if (!pristine && !force) guard()
  }

  await mkdir(path.dirname(absFile), { recursive: true })
  await writeFile(absFile, code)
  return {
    entry: makeEntry(existing, component, relFile, ctx.framework, hash, input),
    status: current === null ? 'installed' : 'updated',
    components: custom.components
  }
}

/** Convenience for the common case: a code response from the server. */
export function writeShaderFile(ctx: ProjectContext, response: CodeResponse, existing: LockEntry | null, lock: LockFile, force: boolean, source?: LockSource): Promise<WriteResult> {
  return writeComponentFile(ctx, {
    id: response.id,
    title: response.title,
    code: response.code,
    files: response.files,
    updatedAt: response.updated_at ?? null,
    source: source ?? existing?.source
  }, existing, lock, force)
}

function makeEntry(existing: LockEntry | null, component: string, file: string, framework: ShadersFramework, hash: string, input: WriteInput): LockEntry {
  const entry: LockEntry = {
    title: input.title,
    component,
    file,
    framework,
    hash,
    updatedAt: input.updatedAt,
    installedAt: existing?.installedAt ?? new Date().toISOString()
  }
  if (input.source) entry.source = input.source
  return entry
}

// ---------------------------------------------------------------------------
// Picking from the connected project
// ---------------------------------------------------------------------------

/** Every shader in the project, following pagination up to MAX_SHADERS. */
export async function listProjectShaders(credentials: Credentials, projectId: string): Promise<{ project: { id: string, title: string }, shaders: ProjectShader[] }> {
  const shaders: ProjectShader[] = []
  let offset: number | null = 0
  let project: { id: string, title: string } | null = null
  while (offset !== null && shaders.length < MAX_SHADERS) {
    let page: ProjectShadersPage
    try {
      page = await api<ProjectShadersPage>(
        `/api/plugin/projects/${encodeURIComponent(projectId)}/shaders`,
        { token: credentials.accessToken, query: { limit: String(PAGE_SIZE), offset: String(offset) } }
      )
    } catch (error) {
      if (error instanceof ApiError && error.isProjectNotFound) throw projectGone(projectId)
      throw error
    }
    project ??= page.project
    shaders.push(...page.shaders)
    offset = page.has_more ? page.next_offset : null
  }
  return { project: project!, shaders }
}

/**
 * No refs given: offer the shaders inside the connected project. Connects one
 * first when the config has no project yet.
 */
async function pickShaders(ctx: ProjectContext, credentials: Credentials, flags: InstallFlags): Promise<string[]> {
  if (!isInteractive(flags) && !flags.all) {
    throw new Error('Usage: npx shaders install <id|preset> [...]  (or --all to install every shader in the connected project)\nShader IDs are shown in the design editor\'s Export Code dialog; presets look like offsets-1.')
  }

  let projectId = ctx.project
  if (!projectId) {
    consola.info('This codebase isn\'t connected to a Shaders project yet')
    const project = await chooseProject(credentials, ctx.detected, flags)
    if (!project) throw new Error('No project selected. Pass --project <id> or run npx shaders connect')
    projectId = project.id
    if (await setConfigField(ctx.configFile, 'project', projectId)) {
      consola.success(`Connected project "${project.title}"`)
    } else {
      consola.warn(`Couldn't update ${path.basename(ctx.configFile)} automatically — add project: '${projectId}' to it`)
    }
  }

  const { project, shaders } = await listProjectShaders(credentials, projectId)
  if (shaders.length === 0) {
    consola.info(`"${project.title}" has no shaders yet. Run npx shaders open to design some in the Shaders app.`)
    return []
  }
  if (flags.all) return shaders.map(s => s.id)

  const picked = await consola.prompt(`Which shaders from "${project.title}" do you want to install?`, {
    type: 'multiselect',
    options: [
      { value: ALL, label: `All ${shaders.length} shaders` },
      ...shaders.map(s => ({ value: s.id, label: s.title || `Shader ${s.id}`, hint: `id ${s.id}` }))
    ],
    required: true,
    cancel: 'symbol'
  })
  if ((picked as unknown) === CANCEL) cancelled()
  const values = (picked as unknown as Array<string | { value: string }>).map(p => typeof p === 'string' ? p : p.value)
  return values.includes(ALL) ? shaders.map(s => s.id) : values
}

// ---------------------------------------------------------------------------
// Install
// ---------------------------------------------------------------------------

const COMPONENT_VERBS: Record<WriteResult['components'][number]['status'], string> = {
  installed: 'added',
  updated: 'updated',
  removed: 'removed (no longer used)',
  kept: 'no longer used, kept because it has local edits'
}

/** One line per custom component file touched, for the install / update report. */
export function componentLines(components: WriteResult['components']): string {
  return components.map(c => `\n  custom component ${c.file}: ${COMPONENT_VERBS[c.status]}`).join('')
}

function report(ctx: ProjectContext, title: string, entry: LockEntry, status: WriteResult['status'], extra?: string, components: WriteResult['components'] = []) {
  const verb = status === 'unchanged' ? 'Already up to date' : status === 'updated' ? 'Updated' : 'Installed'
  consola.success(`${verb} "${title}" → ${entry.file}${componentLines(components)}${extra ? `\n  ${extra}` : ''}\n  ${usageHint(ctx, entry)}`)
}

/** One of the user's own saved shaders. */
async function installShader(ctx: ProjectContext, credentials: Credentials, id: string, lock: LockFile, flags: InstallFlags): Promise<void> {
  const response = await fetchCode(credentials, id, ctx.framework)
  const existing = lock.shaders[id] ?? null
  const { entry, status, components } = await writeShaderFile(ctx, response, existing, lock, flags.force)
  lock.shaders[id] = entry
  await writeLockFile(ctx.detected.dir, lock)

  if (ctx.project && response.kind === 'shader' && response.project_id && response.project_id !== ctx.project) {
    consola.warn(`"${response.title}" belongs to a different Shaders project than this codebase is connected to`)
  }
  report(ctx, response.title, entry, status, undefined, components)
}

/**
 * Pro: clone the preset into the connected project and install the clone
 * (the lock key becomes the new saved shader's id). Otherwise: offer screen.
 */
async function installPreset(ctx: ProjectContext, ref: string, lock: LockFile, flags: InstallFlags, state: Awaited<ReturnType<typeof getSignedInState>>): Promise<Awaited<ReturnType<typeof getSignedInState>>> {
  const preset = await resolvePreset(ref, flags)

  // Already cloned into the project? Re-running is a refresh of that clone,
  // not a second copy in the user's project. (Pending clones — created but
  // never written — are resumed by upgradeToClone below instead.)
  const cloneKey = Object.keys(lock.shaders).find(k => lock.shaders[k].source?.kind === 'preset' && lock.shaders[k].source?.presetId === preset.id && !lock.shaders[k].pending)
  if (cloneKey) {
    const session = state ?? await ensureSignedIn()
    consola.info(`"${preset.title}" is already in your project as shader ${cloneKey} — refreshing it`)
    await installShader(ctx, session.credentials, cloneKey, lock, flags)
    return session
  }

  // Already installed as a preview? Hand off to the same upgrade path update uses.
  const previewKey = Object.keys(lock.shaders).find(k => lock.shaders[k].source?.kind === 'preview' && lock.shaders[k].source?.presetId === preset.id)

  let session = state
  if (!session?.me.isPro) {
    const outcome = await offerPro(preset, session, flags)
    if (outcome.kind === 'preview') {
      const existing = previewKey ? lock.shaders[previewKey] : null
      const { entry, status } = await writeComponentFile(ctx, {
        id: preset.id,
        title: preset.title,
        code: renderPreviewFile(ctx.framework, existing?.component ?? componentNameFromTitle(preset.title, preset.id), preset),
        updatedAt: null,
        source: presetSource(preset, 'preview')
      }, existing, lock, flags.force)
      lock.shaders[preset.id] = entry
      await writeLockFile(ctx.detected.dir, lock)
      report(ctx, preset.title, entry, status, 'Watermarked preview — run `npx shaders update` once you have Pro to swap in the real component')
      return session
    }
    session = { credentials: outcome.credentials, me: outcome.me }
    consola.success(`Signed in${session.me.email ? ` as ${session.me.email}` : ''} — Shaders Pro`)
  }

  const projectId = await ensureProject(ctx, session.credentials, flags)
  await upgradeToClone(ctx, session.credentials, preset, projectId, previewKey ? lock.shaders[previewKey] : null, previewKey, lock, flags.force)
  return session
}

/**
 * Clone `preset` into the project and write the clone's source over
 * `existing` (a preview entry, or nothing). Re-keys the lock from the preset
 * id to the new saved-shader id. Shared with update's preview upgrade.
 *
 * Two phases so a failure between them can't orphan a shader in the user's
 * project: the clone is recorded in the lock as `pending` the moment it
 * exists, and a retry (or `update`) picks that clone up rather than making
 * another. The marker clears only once the file and final entry are written.
 */
export async function upgradeToClone(ctx: ProjectContext, credentials: Credentials, preset: PresetSummary, projectId: string, existing: LockEntry | null, existingKey: string | null | undefined, lock: LockFile, force: boolean): Promise<void> {
  // Phase 1 — find or create the clone, and pin it in the lock immediately.
  let cloneId = Object.keys(lock.shaders).find(k => lock.shaders[k].pending && lock.shaders[k].source?.presetId === preset.id)
  let pending: LockEntry
  if (cloneId) {
    pending = lock.shaders[cloneId]
    consola.info(`Resuming "${pending.title}" — clone ${cloneId} was created earlier but never written`)
  } else {
    const clone = await clonePreset(credentials, preset, projectId)
    cloneId = clone.id
    const placement = existing
      ? { component: existing.component, relFile: existing.file }
      : pickFreeName(ctx, clone.title, clone.id, lock)
    pending = {
      title: clone.title,
      component: placement.component,
      file: placement.relFile,
      framework: ctx.framework,
      // Carry the preview's hash so writeComponentFile's pristine check
      // recognises an untouched preview file — and refuses a hand-edited one
      // unless --force, same as any other update.
      hash: existing?.hash ?? '',
      updatedAt: clone.updated_at,
      installedAt: existing?.installedAt ?? new Date().toISOString(),
      source: presetSource(preset, 'preset'),
      pending: true
    }
    if (existingKey && existingKey !== cloneId) delete lock.shaders[existingKey]
    lock.shaders[cloneId] = pending
    await writeLockFile(ctx.detected.dir, lock)
  }

  // Phase 2 — fetch the clone's source and write the component file.
  const response = await fetchCode(credentials, cloneId, ctx.framework)
  const { entry, status, components } = await writeComponentFile(ctx, {
    id: cloneId,
    title: response.title,
    code: response.code,
    files: response.files,
    updatedAt: response.updated_at ?? pending.updatedAt,
    source: presetSource(preset, 'preset')
  }, pending, lock, force)
  lock.shaders[cloneId] = entry // makeEntry never sets `pending` → marker cleared
  await writeLockFile(ctx.detected.dir, lock)
  report(ctx, response.title, entry, existing ? 'updated' : status,
    `Added to your project as shader ${cloneId} — edit it with \`npx shaders open\``, components)
}

export async function install(refs: string[], flags: InstallFlags): Promise<void> {
  const cwd = process.cwd()
  const presetFirst = refs.length > 0 && refs.every(isPresetRef)

  // First run in this project: walk through connect, then carry on installing.
  // For a preset-only install, connect runs without the sign-in step so a
  // signed-out visitor reaches the offer screen instead of a login wall.
  if (!(await hasProjectConfig(cwd))) {
    if (!isInteractive(flags)) throw new Error(`No shaders.config found. Run npx shaders connect first.`)
    consola.info('Shaders isn\'t set up in this project yet — running connect first\n')
    await init(presetFirst ? { ...flags, auth: false } : flags)
    consola.log('')
  }

  const ctx = await resolveProjectContext(cwd)
  const lock: LockFile = await readLockFile(ctx.detected.dir)

  // Presets decide about auth themselves; everything else needs a session.
  let state = await getSignedInState()
  const needsSession = refs.length === 0 || refs.some(r => !isPresetRef(r))
  if (needsSession && !state) {
    const signedIn = await ensureSignedIn()
    state = signedIn
  }

  if (refs.length === 0) {
    refs = await pickShaders(ctx, state!.credentials, flags)
    if (refs.length === 0) return
  }

  let failures = 0
  for (const ref of refs) {
    try {
      if (isPresetRef(ref)) {
        state = await installPreset(ctx, ref, lock, flags, state)
      } else {
        await installShader(ctx, state!.credentials, ref, lock, flags)
      }
    } catch (error) {
      failures++
      consola.error(`${ref}: ${error instanceof Error ? error.message : String(error)}`)
    }
  }

  if (failures === refs.length) process.exit(1)
}
