// `npx shaders mcp` (alias install-mcp) / `uninstall-mcp` — thin wrapper around add-mcp's
// programmatic API, which owns the per-agent config formats and file locations.
import {
  agents,
  detectGlobalAgents,
  detectProjectAgents,
  getAgentTypes,
  removeServer,
  upsertServer,
  type AgentType
} from 'add-mcp'
import * as p from '@clack/prompts'
import { fail } from './fail'


export const MCP_URL = 'https://shaders.com/mcp'
const SERVER_NAME = 'shaders'
export const MCP_DOCS_URL = 'https://shaders.com/docs/guide/mcp'

export interface McpFlags {
  global: boolean
  yes: boolean
  agents: string[]
}

/**
 * The Shaders MCP server is remote (HTTP). Agents whose config file only
 * accepts local stdio servers (Claude Desktop) can't use it, so install keeps
 * them out of detection, selection and -a validation rather than silently
 * writing an entry they'll ignore — add-mcp's programmatic upsertServer
 * doesn't check. Uninstall still sweeps every agent so stale entries go.
 */
function supportsHttp(type: AgentType): boolean {
  return agents[type].supportedTransports.includes('http')
}

export function getHttpAgentTypes(): AgentType[] {
  return getAgentTypes().filter(supportsHttp)
}

function validateAgents(names: string[], { requireHttp }: { requireHttp: boolean }): AgentType[] {
  const known = new Set(getAgentTypes())
  const allowed = requireHttp ? getHttpAgentTypes() : getAgentTypes()
  const allowedSet = new Set(allowed)
  for (const name of names) {
    if (!known.has(name as AgentType)) {
      fail(`Unknown agent "${name}". Supported agents: ${allowed.join(', ')}`)
    }
    if (!allowedSet.has(name as AgentType)) {
      const agent = agents[name as AgentType]
      fail(`${agent.displayName}: ${agent.unsupportedTransportMessage ?? 'this agent does not support remote (HTTP) MCP servers.'}`)
    }
  }
  return names as AgentType[]
}

/**
 * Resolve which agents to install to. With -a the user picks explicitly.
 * Otherwise agents are auto-detected (project configs in cwd by default,
 * global installs with -g or when nothing is found in the project), and an
 * interactive multiselect confirms the selection unless -y / non-TTY.
 */
async function resolveTargets(flags: McpFlags, cwd: string): Promise<{ targets: AgentType[], scope: 'project' | 'global' }> {
  if (flags.agents.length > 0) {
    return {
      targets: validateAgents(flags.agents, { requireHttp: true }),
      scope: flags.global ? 'global' : 'project'
    }
  }

  const projectDetected = flags.global ? [] : detectProjectAgents(cwd).filter(supportsHttp)
  const scope: 'project' | 'global' = flags.global || projectDetected.length === 0 ? 'global' : 'project'
  const detected = scope === 'project' ? projectDetected : (await detectGlobalAgents()).filter(supportsHttp)

  const interactive = !flags.yes && process.stdout.isTTY && process.stdin.isTTY
  if (!interactive) {
    if (detected.length === 0) {
      fail(`No coding agents detected. Pass one explicitly, e.g. npx shaders mcp -a cursor\nSupported agents: ${getHttpAgentTypes().join(', ')}`)
    }
    return { targets: detected, scope }
  }

  const selection = await p.multiselect({
    message: `Install the Shaders MCP server to which agents? (${scope} config)`,
    options: getHttpAgentTypes().map(type => ({
      value: type,
      label: agents[type].displayName,
      hint: scope === 'project' && agents[type].localConfigPath
        ? agents[type].localConfigPath
        : agents[type].configPath.replace(/^\/Users\/[^/]+|^\/home\/[^/]+|^C:\\Users\\[^\\]+/, '~')
    })),
    initialValues: detected,
    required: true
  })

  if (p.isCancel(selection)) {
    console.log('Cancelled — nothing was changed.')
    process.exit(0)
  }

  return { targets: selection as AgentType[], scope }
}

export async function installMcp(flags: McpFlags) {
  const cwd = process.cwd()
  const { targets, scope } = await resolveTargets(flags, cwd)

  let failures = 0
  for (const target of targets) {
    // Agents without project-level config (Claude Desktop, Windsurf, …) fall
    // back to their global config instead of erroring.
    const local = scope === 'project' && !!agents[target].localConfigPath
    const result = upsertServer(target, SERVER_NAME, { type: 'http', url: MCP_URL }, { local, cwd })

    if (result.success) {
      console.log(`✓ ${agents[target].displayName} — ${result.path}`)
    } else {
      failures++
      console.error(`✗ ${agents[target].displayName} — ${result.error}`)
    }
  }

  if (failures === targets.length) process.exit(1)

  console.log(`
Done! Open your agent and connect to the "${SERVER_NAME}" MCP server — you'll be
prompted to sign in to your Shaders account the first time it connects.

Docs: ${MCP_DOCS_URL}`)
}

export async function uninstallMcp(flags: McpFlags) {
  const cwd = process.cwd()
  const targets = flags.agents.length > 0 ? validateAgents(flags.agents, { requireHttp: false }) : getAgentTypes()

  let removedCount = 0
  let attempts = 0
  let failures = 0
  for (const target of targets) {
    // With -g only touch global config; otherwise sweep project + global
    const scopes = flags.global || !agents[target].localConfigPath ? [false] : [true, false]
    for (const local of scopes) {
      attempts++
      const result = removeServer(target, SERVER_NAME, { local, cwd })
      if (result.success && result.removed) {
        removedCount++
        console.log(`✓ Removed from ${agents[target].displayName} — ${result.path}`)
      } else if (!result.success) {
        failures++
        console.error(`✗ ${agents[target].displayName} — ${result.error}`)
      }
    }
  }

  if (attempts > 0 && failures === attempts) process.exit(1)

  if (removedCount === 0 && failures === 0) {
    console.log('No Shaders MCP server entries found.')
  }
}

