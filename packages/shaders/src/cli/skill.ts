// `npx shaders skill` — install the Shaders agent skill into the project's coding agents.
//
// Thin wrapper around the `skills` CLI (https://skills.sh), which owns the per-agent
// skill directories, symlinking and updates, so `npx skills update` keeps the skill
// current. The skill itself lives in this repo at skills/shaders/SKILL.md and is
// fetched from GitHub at install time.
import { spawn } from 'node:child_process'

export const SKILL_SOURCE = 'shader-effects-inc/shaders'
export const SKILL_NAME = 'shaders'
export const SKILL_DOCS_URL = 'https://shaders.com/docs/guide/agent-skill'

export interface SkillFlags {
  global: boolean
  yes: boolean
  /** Agent names as the skills CLI knows them (claude-code, cursor, codex, …) */
  agents: string[]
  /** Project root (where the agent folders live); defaults to the current directory */
  cwd?: string
}

export function skillInstallArgs(flags: SkillFlags): string[] {
  const args = ['--yes', 'skills@latest', 'add', SKILL_SOURCE, '--skill', SKILL_NAME]
  if (flags.global) args.push('--global')
  for (const agent of flags.agents) args.push('--agent', agent)
  if (flags.yes) args.push('--yes')
  return args
}

/** Runs `npx skills add …` with the terminal attached, so its prompts work as usual. */
export function installSkill(flags: SkillFlags): Promise<void> {
  return new Promise((resolve, reject) => {
    const windows = process.platform === 'win32'
    const child = spawn(windows ? 'npx.cmd' : 'npx', skillInstallArgs(flags), { stdio: 'inherit', shell: windows, cwd: flags.cwd })
    child.on('error', reject)
    child.on('exit', (code) => {
      if (code === 0) resolve()
      else reject(new Error(`skills add exited with code ${code ?? 'null'}`))
    })
  })
}
