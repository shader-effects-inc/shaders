// `npx shaders connect` — set up Shaders in a downstream project:
//   1. detect the framework from package.json
//   2. install the `shaders` package with the project's package manager
//   3. write shaders.config.ts
//   4. sign in and connect the codebase to a shaders.com project
//   5. add the Shaders agent skill to the project's coding agents
import path from 'node:path'
import { consola } from 'consola'
import { addDependency, detectPackageManager } from 'nypm'
import { api } from './api'
import { ensureSignedIn, type Credentials } from './auth'
import { findConfigFile, setConfigField, writeConfigFile } from './configFile'
import { defaultOutDir, describeDetection, detectProject, LIBRARY_LABELS, type DetectedProject, type Library } from './detect'
import { installSkill, SKILL_DOCS_URL } from './skill'

export interface InitFlags {
  yes: boolean
  auth: boolean
  install: boolean
  skill: boolean
  project?: string
  framework?: string
}

export interface ProjectSummary {
  id: string
  title: string
  updated_at: string
  shader_count: number
}

const CANCEL = Symbol.for('cancel')
const LIBRARIES = Object.keys(LIBRARY_LABELS) as Library[]

export function isInteractive(flags: { yes: boolean }): boolean {
  return !flags.yes && !!process.stdout.isTTY && !!process.stdin.isTTY
}

export function cancelled(): never {
  consola.log('Cancelled.')
  process.exit(0)
}

async function resolveLibrary(detected: DetectedProject, flags: InitFlags): Promise<Library> {
  if (flags.framework) {
    if (!LIBRARIES.includes(flags.framework as Library)) {
      throw new Error(`Unknown framework "${flags.framework}". Use one of: ${LIBRARIES.join(', ')}`)
    }
    return flags.framework as Library
  }
  if (detected.library) return detected.library

  if (!isInteractive(flags)) {
    throw new Error(`Couldn't detect a UI framework in ${path.join(detected.dir, 'package.json')}. Pass one with --framework <${LIBRARIES.join('|')}>`)
  }
  const choice = await consola.prompt('Which framework does this project use?', {
    type: 'select',
    options: LIBRARIES.map(value => ({ value, label: LIBRARY_LABELS[value] })),
    cancel: 'symbol'
  })
  if ((choice as unknown) === CANCEL) cancelled()
  return choice as Library
}

async function installPackage(detected: DetectedProject, flags: InitFlags): Promise<void> {
  if (detected.shadersVersion) {
    consola.success(`shaders already installed (${detected.shadersVersion})`)
    return
  }
  if (!flags.install) {
    consola.info('Skipped installing shaders (--no-install)')
    return
  }
  // No lockfile yet (fresh scaffold) → nypm can't detect; fall back to npm.
  const pm = (await detectPackageManager(detected.dir))?.name ?? 'npm'
  consola.start(`Installing shaders with ${pm}…`)
  try {
    await addDependency('shaders', { cwd: detected.dir, silent: true, packageManager: pm })
  } catch (error) {
    throw new Error(`Failed to install shaders: ${error instanceof Error ? error.message : String(error)}\nInstall it manually, then re-run npx shaders connect`)
  }
  consola.success('Installed shaders')
}

function relative(dir: string, file: string): string {
  return path.relative(dir, file) || path.basename(file)
}

export async function chooseProject(credentials: Credentials, detected: DetectedProject, flags: InitFlags): Promise<ProjectSummary | null> {
  if (flags.project) {
    const { projects } = await api<{ projects: ProjectSummary[] }>('/api/plugin/projects', { token: credentials.accessToken, query: { id: flags.project, limit: '1' } })
    const found = projects.find(p => p.id === flags.project)
    if (!found) throw new Error(`Project ${flags.project} isn't in your account — it may have been deleted. Run npx shaders connect without --project to pick one.`)
    return found
  }

  if (!isInteractive(flags)) {
    consola.info('Skipped connecting a project — pass --project <id> to connect non-interactively')
    return null
  }

  const { projects } = await api<{ projects: ProjectSummary[] }>('/api/plugin/projects', { token: credentials.accessToken, query: { limit: '100' } })

  const NEW = '__new__'
  const options = [
    { value: NEW, label: projects.length ? 'Create a new project' : 'Create a new project (you have none yet)' },
    ...projects.map(p => ({
      value: p.id,
      label: p.title || 'Untitled Project',
      hint: `${p.shader_count} shader${p.shader_count === 1 ? '' : 's'}`
    }))
  ]
  const choice = await consola.prompt('Which project should this codebase connect to?', {
    type: 'select',
    options,
    cancel: 'symbol'
  })
  if ((choice as unknown) === CANCEL) cancelled()

  if (choice !== NEW) {
    return projects.find(p => p.id === choice) ?? null
  }

  const defaultTitle = typeof detected.pkg.name === 'string' && detected.pkg.name
    ? detected.pkg.name.replace(/^@[^/]+\//, '')
    : path.basename(detected.dir)
  const title = await consola.prompt('Project name', { type: 'text', default: defaultTitle, placeholder: defaultTitle, cancel: 'symbol' })
  if ((title as unknown) === CANCEL) cancelled()

  const created = await api<{ id: string, title: string }>('/api/plugin/projects', {
    method: 'POST',
    token: credentials.accessToken,
    body: { title: (title as string) || defaultTitle }
  })
  return { id: created.id, title: created.title, updated_at: new Date().toISOString(), shader_count: 0 }
}

export interface Setup {
  detected: DetectedProject
  library: Library
  /** Path of shaders.config, or null when none exists yet (connect/the wizard write one) */
  configFile: string | null
}

/**
 * The anonymous half of `connect`: detect the framework, install the package,
 * find an existing config. Nothing here needs an account, so the wizard runs
 * it on first launch before asking anything.
 */
export async function ensureSetup(flags: InitFlags, { quiet = false } = {}): Promise<Setup> {
  const cwd = process.cwd()
  const detected = await detectProject(cwd)
  if (!detected) {
    throw new Error(`No package.json found in ${cwd} or its parents. Run npx shaders from inside your project.`)
  }

  const library = await resolveLibrary(detected, flags)
  if (!quiet) consola.log(`Detected: ${describeDetection(detected.framework, library)}`)

  await installPackage(detected, flags)

  const configFile = await findConfigFile(detected.dir)
  if (configFile && !quiet) {
    consola.success(`Found ${relative(detected.dir, configFile)}`)
  }
  return { detected, library, configFile }
}

/** Write a fresh shaders.config (no project yet) and say where components will land. */
export async function writeInitialConfig(setup: Setup, projectId: string | null): Promise<string> {
  const outDir = defaultOutDir(setup.detected.dir, setup.detected.framework)
  const configFile = await writeConfigFile(setup.detected.dir, { framework: setup.library, project: projectId ?? undefined, outDir }, setup.detected.usesTypeScript)
  consola.success(`Created ${relative(setup.detected.dir, configFile)} (components will install to ${outDir}/)`)
  return configFile
}

export async function init(flags: InitFlags): Promise<void> {
  const setup = await ensureSetup(flags)
  const { detected } = setup
  let { configFile } = setup

  let projectId: string | null = null
  let projectTitle: string | null = null
  if (flags.auth) {
    const { credentials, me } = await ensureSignedIn()
    consola.success(`Signed in${me.email ? ` as ${me.email}` : ''}`)
    const project = await chooseProject(credentials, detected, flags)
    if (project) {
      projectId = project.id
      projectTitle = project.title
    }
  } else {
    consola.info('Skipped sign-in (--no-auth)')
  }

  if (!configFile) {
    configFile = await writeInitialConfig(setup, projectId)
  } else if (projectId) {
    const updated = await setConfigField(configFile, 'project', projectId)
    if (!updated) {
      consola.warn(`Couldn't update ${relative(detected.dir, configFile)} automatically — add project: '${projectId}' to it`)
    }
  }

  if (projectId) {
    consola.success(`Connected project to Shaders${projectTitle ? ` (${projectTitle})` : ''}`)
  }

  // Best effort: the skill is a convenience for the user's coding agents, so a
  // failure here (offline, npx unavailable) must not undo a successful connect.
  if (flags.skill) {
    consola.start('Adding the Shaders agent skill to your coding agents…')
    try {
      await installSkill({ global: false, yes: flags.yes, agents: [], cwd: detected.dir })
    } catch {
      consola.warn(`Couldn't add the agent skill. Run npx shaders skill to retry (${SKILL_DOCS_URL})`)
    }
  } else {
    consola.info('Skipped the agent skill (--no-skill)')
  }
}
