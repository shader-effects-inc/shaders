// shaders.lock.json — what `install` wrote and from where, so `update` knows
// which files to refresh and can tell a locally edited file from a pristine
// one. Lives next to shaders.config; plain JSON, safe to commit.
import { createHash } from 'node:crypto'
import { readFile, writeFile } from 'node:fs/promises'
import path from 'node:path'
import type { ShadersFramework } from '../config'

export const LOCK_FILENAME = 'shaders.lock.json'

/**
 * Where an entry came from, when it wasn't a plain saved-shader install.
 *  - `preset`: cloned from a curated preset into the user's project; the
 *    entry's key is the new saved shader's numeric id.
 *  - `preview`: the watermarked <Preview> placeholder written for a
 *    signed-out / non-Pro user; the entry's key is the preset id. `update`
 *    turns it into a `preset` entry once the user has Pro.
 */
export interface LockSource {
  kind: 'preset' | 'preview'
  presetId: string
  slug: string | null
}

export interface LockEntry {
  title: string
  component: string
  /** Path relative to the project root */
  file: string
  framework: ShadersFramework
  /** sha256 of the file contents as written by the CLI */
  hash: string
  /** Server-side updated_at of the source shader at install/update time */
  updatedAt: string | null
  installedAt: string
  source?: LockSource
  /**
   * Set between "preset cloned into the project" and "component file
   * written". A retry after a failure in between reuses this clone instead of
   * creating another; `update` resumes it.
   */
  pending?: true
}

/**
 * A custom component file (`custom/<Name>.ts`, a defineShader definition) that
 * installed shaders import. Keyed in the lock by its layer type
 * (`Custom:Halo_1a2b3c4d`): identical components share one file.
 */
export interface ComponentLockEntry {
  /** Path relative to the project root */
  file: string
  /** The path the server suggested, relative to the shader's folder (`custom/Halo.ts`) */
  suggested: string
  /** sha256 of the file contents as written by the CLI */
  hash: string
  /** Lock keys of the shaders that import it */
  usedBy: string[]
}

export interface LockFile {
  /** 2 once any custom component is installed (older CLIs refuse it rather than drop `components`) */
  version: 1 | 2
  shaders: Record<string, LockEntry>
  components: Record<string, ComponentLockEntry>
}

export function hashContent(content: string): string {
  return 'sha256:' + createHash('sha256').update(content).digest('hex')
}

const ENTRY_STRING_FIELDS = ['title', 'component', 'file', 'framework', 'hash', 'installedAt'] as const

function isLockSource(value: unknown): value is LockSource {
  if (!value || typeof value !== 'object') return false
  const src = value as Record<string, unknown>
  return (src.kind === 'preset' || src.kind === 'preview')
    && typeof src.presetId === 'string'
    && (src.slug === null || typeof src.slug === 'string')
}

function isComponentLockEntry(value: unknown): value is ComponentLockEntry {
  if (!value || typeof value !== 'object') return false
  const entry = value as Record<string, unknown>
  return typeof entry.file === 'string' && typeof entry.suggested === 'string' && typeof entry.hash === 'string'
    && Array.isArray(entry.usedBy) && entry.usedBy.every(key => typeof key === 'string')
}

function isLockEntry(value: unknown): value is LockEntry {
  if (!value || typeof value !== 'object') return false
  const entry = value as Record<string, unknown>
  return ENTRY_STRING_FIELDS.every(key => typeof entry[key] === 'string')
    && (entry.updatedAt === null || typeof entry.updatedAt === 'string')
    && (entry.source === undefined || isLockSource(entry.source))
    && (entry.pending === undefined || entry.pending === true)
}

/**
 * A missing lock file means "nothing installed". Anything else that goes
 * wrong — unreadable, malformed JSON, unknown version, bad entries — throws,
 * because silently treating it as empty would let the next write discard
 * every existing entry.
 */
export async function readLockFile(dir: string): Promise<LockFile> {
  const file = path.join(dir, LOCK_FILENAME)
  let raw: string
  try {
    raw = await readFile(file, 'utf8')
  } catch (error) {
    if ((error as NodeJS.ErrnoException).code === 'ENOENT') return { version: 1, shaders: {}, components: {} }
    throw error
  }

  let parsed: unknown
  try {
    parsed = JSON.parse(raw)
  } catch {
    throw new Error(`${LOCK_FILENAME} is not valid JSON — fix or delete it and re-run`)
  }
  if (!parsed || typeof parsed !== 'object') throw new Error(`${LOCK_FILENAME} is malformed — fix or delete it and re-run`)
  const lock = parsed as Partial<LockFile>
  if (lock.version !== 1 && lock.version !== 2) throw new Error(`${LOCK_FILENAME} has unsupported version ${String(lock.version)} — update the shaders CLI`)
  if (!lock.shaders || typeof lock.shaders !== 'object' || Array.isArray(lock.shaders)) {
    throw new Error(`${LOCK_FILENAME} is malformed (missing "shaders") — fix or delete it and re-run`)
  }
  for (const [id, entry] of Object.entries(lock.shaders)) {
    if (!isLockEntry(entry)) throw new Error(`${LOCK_FILENAME} has an invalid entry for "${id}" — fix or delete it and re-run`)
  }
  const components = lock.version === 2 ? lock.components ?? {} : {}
  if (typeof components !== 'object' || Array.isArray(components)) {
    throw new Error(`${LOCK_FILENAME} is malformed ("components" is not an object) — fix or delete it and re-run`)
  }
  for (const [type, entry] of Object.entries(components)) {
    if (!isComponentLockEntry(entry)) throw new Error(`${LOCK_FILENAME} has an invalid component entry for "${type}" — fix or delete it and re-run`)
  }
  return { version: lock.version, shaders: lock.shaders, components }
}

export async function writeLockFile(dir: string, lock: LockFile): Promise<void> {
  const sortedEntries = <T>(record: Record<string, T>) => Object.fromEntries(Object.entries(record).sort(([a], [b]) => a.localeCompare(b)))
  const hasComponents = Object.keys(lock.components).length > 0
  // Stays version 1 (readable by older CLIs) until a custom component is installed
  const sorted = hasComponents
    ? { version: 2, shaders: sortedEntries(lock.shaders), components: sortedEntries(lock.components) }
    : { version: 1, shaders: sortedEntries(lock.shaders) }
  await writeFile(path.join(dir, LOCK_FILENAME), JSON.stringify(sorted, null, 2) + '\n')
}
