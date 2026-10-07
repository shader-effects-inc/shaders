// Bare `npx shaders` in a terminal: a router over the other commands that
// reads the project's state first and only asks what state can't answer.
// Everything that can run without an account does (package install, config,
// search, skill, MCP); sign-in happens inside the actions that need it.
// Non-interactive callers (CI, agents) never get here — cli.ts prints help.
import { existsSync } from 'node:fs'
import { homedir } from 'node:os'
import path from 'node:path'
import { consola } from 'consola'
import { listInstalledServers } from 'add-mcp'
import { getSignedInState, login, type Credentials, type Me } from './auth'
import { findConfigFile, readConfig, setConfigField, type ConfigValues } from './configFile'
import { ensureSetup, cancelled, chooseProject, writeInitialConfig, type InitFlags, type Setup } from './init'
import { install, listProjectShaders, type InstallFlags } from './install'
import { installMcp } from './installMcp'
import { open } from './open'
import { readLockFile } from './lockFile'
import { printResults, searchPresets } from './search'
import { installSkill, SKILL_NAME } from './skill'
import { update } from './update'

const CANCEL = Symbol.for('cancel')
const MCP_SERVER_NAME = 'shaders'

type Scope = 'project' | 'global' | null

interface State {
  setup: Setup
  config: ConfigValues | null
  installed: number
  /** Installed shaders whose source changed in the editor; null when unknown (signed out) */
  changed: number | null
  signedIn: { credentials: Credentials, me: Me } | null
  projectTitle: string | null
  skill: Scope
  mcp: Scope
}

type Action = 'update' | 'install' | 'search' | 'editor' | 'skill' | 'mcp' | 'done'

const WIZARD_FLAGS: InstallFlags = { yes: false, auth: true, install: true, skill: false, force: false, all: false }

// ---------------------------------------------------------------------------
// State
// ---------------------------------------------------------------------------

function skillScope(projectDir: string): Scope {
  const dirs: Array<[string, Scope]> = [
    [path.join(projectDir, '.agents/skills', SKILL_NAME), 'project'],
    [path.join(projectDir, '.claude/skills', SKILL_NAME), 'project'],
    [path.join(projectDir, '.cursor/skills', SKILL_NAME), 'project'],
    [path.join(homedir(), '.agents/skills', SKILL_NAME), 'global'],
    [path.join(homedir(), '.claude/skills', SKILL_NAME), 'global']
  ]
  return dirs.find(([dir]) => existsSync(dir))?.[1] ?? null
}

async function mcpScope(projectDir: string): Promise<Scope> {
  try {
    const local = await listInstalledServers({ cwd: projectDir })
    if (local.some(agent => agent.servers.some(s => s.serverName === MCP_SERVER_NAME && s.scope === 'local'))) return 'project'
    const global = await listInstalledServers({ cwd: projectDir, global: true })
    if (global.some(agent => agent.servers.some(s => s.serverName === MCP_SERVER_NAME))) return 'global'
  } catch {
    // Unreadable agent configs are the same as none for the header's purposes
  }
  return null
}

async function readState(setup: Setup): Promise<State> {
  const projectDir = setup.detected.dir
  const configFile = await findConfigFile(projectDir)
  const config = configFile ? await readConfig(configFile) : null
  const lock = await readLockFile(projectDir)
  const installed = Object.keys(lock.shaders).length

  let signedIn: State['signedIn'] = null
  try {
    signedIn = await getSignedInState()
  } catch {
    signedIn = null
  }

  let changed: number | null = null
  let projectTitle: string | null = null
  if (signedIn && config?.project) {
    try {
      const { project, shaders } = await listProjectShaders(signedIn.credentials, config.project)
      projectTitle = project.title
      changed = shaders.filter((s) => {
        const entry = lock.shaders[s.id]
        return entry && entry.updatedAt && s.updated_at && s.updated_at > entry.updatedAt
      }).length
    } catch {
      // Project gone or offline: the header just says "connected"
    }
  }

  return {
    setup: { ...setup, configFile },
    config,
    installed,
    changed,
    signedIn,
    projectTitle,
    skill: skillScope(projectDir),
    mcp: await mcpScope(projectDir)
  }
}

function header(state: State): string[] {
  const { setup, config } = state
  const framework = setup.detected.framework && setup.detected.framework !== setup.library
    ? `${setup.detected.framework} + ${setup.library}`
    : setup.library
  const first = [`Shaders · ${framework[0]!.toUpperCase()}${framework.slice(1)}`]
  if (config?.project) first.push(state.projectTitle ? `connected to "${state.projectTitle}"` : 'connected')
  else first.push('not connected')

  const second: string[] = []
  if (state.installed) second.push(`${state.installed} installed`)
  if (state.changed) second.push(`${state.changed} changed in the editor`)
  if (state.skill) second.push(`skill ✔${state.skill === 'global' ? ' (global)' : ''}`)
  if (state.mcp) second.push(`MCP ✔${state.mcp === 'global' ? ' (global)' : ''}`)
  if (state.signedIn?.me.email) second.push(state.signedIn.me.email)

  return second.length ? [first.join(' · '), second.join(' · ')] : [first.join(' · ')]
}

// ---------------------------------------------------------------------------
// Actions
// ---------------------------------------------------------------------------

/**
 * Sign in (or sign up) and connect a project, for the actions that need one.
 * Returns the project id. Stored credentials skip the account question.
 */
async function ensureConnected(state: State): Promise<string> {
  // Already connected: opening the editor needs no token (the site signs you in).
  if (state.config?.project) return state.config.project

  let signedIn = state.signedIn
  if (!signedIn) {
    const choice = await consola.prompt('Which account?', {
      type: 'select',
      options: [
        { value: 'sign-in', label: 'Sign in to your Shaders account' },
        { value: 'sign-up', label: 'Create a free account' }
      ],
      cancel: 'symbol'
    })
    if ((choice as unknown) === CANCEL) cancelled()
    const credentials = await login(choice === 'sign-up' ? { screen: 'sign-up' } : {})
    const refreshed = await getSignedInState()
    if (!refreshed) throw new Error('Signed in, but the account could not be verified. Run npx shaders login and try again.')
    signedIn = refreshed
    consola.success(`Signed in${signedIn.me.email ? ` as ${signedIn.me.email}` : ''}`)
    void credentials
  }

  const project = await chooseProject(signedIn.credentials, state.setup.detected, WIZARD_FLAGS)
  if (!project) cancelled()

  if (state.setup.configFile) {
    if (!(await setConfigField(state.setup.configFile, 'project', project.id))) {
      throw new Error(`Couldn't update your shaders config automatically — add project: '${project.id}' to it, then run npx shaders open`)
    }
  } else {
    await writeInitialConfig(state.setup, project.id)
  }
  consola.success(`Connected to "${project.title}"`)
  return project.id
}

async function findPreset(): Promise<void> {
  for (;;) {
    const query = await consola.prompt('Describe the look', { type: 'text', placeholder: 'liquid chrome hero background', cancel: 'symbol' })
    if ((query as unknown) === CANCEL) cancelled()
    const text = String(query ?? '').trim()
    if (!text) return

    consola.start('Searching the library…')
    const { results } = await searchPresets(text)
    if (!results.length) {
      consola.info(`No presets match "${text}". Try different words.`)
      continue
    }
    consola.success(`${results.length} preset${results.length === 1 ? '' : 's'} for "${text}"`)
    printResults(text, results)

    const pick = await consola.prompt('Install one?', {
      type: 'select',
      options: [
        ...results.map(r => ({
          value: r.slug ?? r.id,
          label: r.slug ?? r.id,
          hint: r.similarity != null ? `${Math.round(r.similarity * 100)}% match` : undefined
        })),
        { value: '__again', label: 'Search again' },
        { value: '__back', label: 'Not now' }
      ],
      cancel: 'symbol'
    })
    if ((pick as unknown) === CANCEL) cancelled()
    if (pick === '__back') return
    if (pick === '__again') continue

    await install([pick as string], WIZARD_FLAGS)
    return
  }
}

function buildMenu(state: State): Array<{ value: Action, label: string, hint?: string }> {
  const connected = !!state.config?.project
  const menu: Array<{ value: Action, label: string, hint?: string }> = []
  if (connected && state.changed) {
    menu.push({ value: 'update', label: `Update ${state.changed} changed shader${state.changed === 1 ? '' : 's'}`, hint: 'pull in what changed in the editor' })
  }
  menu.push({ value: 'search', label: 'Find a preset to drop in', hint: 'search the library by describing the look' })
  if (connected) {
    menu.push({ value: 'install', label: 'Install a shader from your project', hint: 'the ones you designed in the editor' })
  }
  menu.push({ value: 'editor', label: 'Open the design editor', hint: connected ? undefined : 'signs you in and connects this codebase to a project' })
  if (!state.skill) menu.push({ value: 'skill', label: 'Install the agent skill', hint: 'teaches Claude Code, Cursor, Codex and others how to build with Shaders' })
  if (!state.mcp) menu.push({ value: 'mcp', label: 'Install the MCP', hint: 'lets your agent search presets and read your shaders' })
  if (connected && state.installed && !state.changed) {
    menu.push({ value: 'update', label: 'Check for updates', hint: `${state.installed} installed` })
  }
  menu.push({ value: 'done', label: 'Nothing right now' })
  return menu
}

async function runAction(action: Action, state: State): Promise<void> {
  switch (action) {
    case 'update':
      await update([], WIZARD_FLAGS)
      break
    case 'install':
      await install([], WIZARD_FLAGS)
      break
    case 'search':
      await findPreset()
      break
    case 'editor':
      await ensureConnected(state)
      await open()
      break
    case 'skill':
      await installSkill({ global: false, yes: false, agents: [], cwd: state.setup.detected.dir })
      break
    case 'mcp':
      await installMcp({ global: false, yes: false, agents: [] })
      break
    case 'done':
      break
  }
}

// ---------------------------------------------------------------------------
// Entry
// ---------------------------------------------------------------------------

export async function runWizard(flags: InitFlags): Promise<void> {
  // First run: install the package and write the config, no questions asked.
  // Repeat runs: the same call is a no-op that just reads what's there.
  const setup = await ensureSetup(flags, { quiet: true })
  if (!setup.configFile) {
    setup.configFile = await writeInitialConfig(setup, null)
  }

  let state = await readState(setup)
  for (;;) {
    consola.log('')
    for (const line of header(state)) consola.log(`  ${line}`)
    consola.log('')

    const action = await consola.prompt('What do you want to ship?', {
      type: 'select',
      options: buildMenu(state),
      cancel: 'symbol'
    })
    if ((action as unknown) === CANCEL || action === 'done') {
      consola.log('')
      return
    }

    await runAction(action as Action, state)
    state = await readState(setup)
  }
}
