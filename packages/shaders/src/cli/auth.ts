// Sign-in for the CLI. Reuses the browser + poll flow built for the Framer
// plugin (/api/plugin-auth/*): we ask the server for a Clerk authorize URL,
// open it in the user's browser, and poll until the redirect handler has
// stored the tokens. Tokens are then kept in ~/.shaders/credentials.json so
// subsequent commands don't prompt again.
import { spawn } from 'node:child_process'
import { chmod, mkdir, readFile, rm, writeFile } from 'node:fs/promises'
import { homedir } from 'node:os'
import path from 'node:path'
import { consola } from 'consola'
import { api, ApiError, getApiUrl } from './api'

const POLL_INTERVAL_MS = 2_000
const POLL_TIMEOUT_MS = 10 * 60_000 // matches the server-side session TTL
const REFRESH_SKEW_MS = 60_000

export interface Credentials {
  host: string
  accessToken: string
  refreshToken?: string
  /** Epoch ms; undefined when the provider didn't say */
  expiresAt?: number
}

export interface Me {
  signedIn: boolean
  isPro: boolean
  hasCore: boolean
  email: string | null
}

interface TokenPayload {
  access_token: string
  refresh_token?: string
  expires_in?: number
}

function credentialsDir(): string {
  return process.env.SHADERS_HOME || path.join(homedir(), '.shaders')
}

function credentialsPath(): string {
  return path.join(credentialsDir(), 'credentials.json')
}

/**
 * `SHADERS_API_KEY` — a personal API key from shaders.com (Settings → API
 * keys) — stands in for the browser sign-in. It's the path for CI and for
 * agents with no browser: the server's Bearer verifier accepts API keys and
 * OAuth tokens alike. When set it wins over stored credentials and is never
 * written to disk or refreshed.
 */
export function apiKeyCredentials(): Credentials | null {
  const key = process.env.SHADERS_API_KEY?.trim()
  return key ? { host: getApiUrl(), accessToken: key } : null
}

export async function loadCredentials(): Promise<Credentials | null> {
  const fromEnv = apiKeyCredentials()
  if (fromEnv) return fromEnv
  try {
    const raw = JSON.parse(await readFile(credentialsPath(), 'utf8')) as Partial<Credentials>
    if (typeof raw.accessToken !== 'string' || raw.host !== getApiUrl()) return null
    return raw as Credentials
  } catch {
    return null
  }
}

async function saveCredentials(credentials: Credentials): Promise<void> {
  await mkdir(credentialsDir(), { recursive: true, mode: 0o700 })
  await writeFile(credentialsPath(), JSON.stringify(credentials, null, 2) + '\n', { mode: 0o600 })
  await chmod(credentialsPath(), 0o600).catch(() => {})
}

export async function clearCredentials(): Promise<boolean> {
  if (apiKeyCredentials()) {
    consola.info('SHADERS_API_KEY is set in this environment — unset it to sign out of that key')
  }
  let existed = true
  try {
    await readFile(credentialsPath())
  } catch {
    existed = false
  }
  await rm(credentialsPath(), { force: true })
  return existed
}

function fromTokens(tokens: TokenPayload): Credentials {
  return {
    host: getApiUrl(),
    accessToken: tokens.access_token,
    refreshToken: tokens.refresh_token,
    expiresAt: tokens.expires_in ? Date.now() + tokens.expires_in * 1000 : undefined
  }
}

export function openBrowser(url: string): void {
  const [command, args] = process.platform === 'darwin'
    ? ['open', [url]]
    : process.platform === 'win32'
      ? ['cmd', ['/c', 'start', '', url.replace(/&/g, '^&')]]
      : ['xdg-open', [url]]
  try {
    const child = spawn(command, args, { stdio: 'ignore', detached: true })
    child.on('error', () => {})
    child.unref()
  } catch {
    // Non-fatal — the URL is printed for the user to open manually
  }
}

const sleep = (ms: number) => new Promise(resolve => setTimeout(resolve, ms))

export interface LoginOptions {
  /** Land on the account sign-up page first (new users), then continue the same OAuth flow. */
  screen?: 'sign-up'
}

/** Full interactive sign-in: browser round-trip + poll, then persist. */
export async function login(options: LoginOptions = {}): Promise<Credentials> {
  if (apiKeyCredentials()) {
    throw new Error('SHADERS_API_KEY is set but was rejected by Shaders — check the key is valid and belongs to the right account, or unset it to sign in with your browser')
  }
  const { url, readKey } = await api<{ url: string, readKey: string }>('/api/plugin-auth/authorize', { method: 'POST', body: options.screen ? { screen: options.screen } : undefined })

  consola.info(`Opening your browser to ${options.screen === 'sign-up' ? 'create your Shaders account' : 'sign in to Shaders'}…\n  If it doesn't open, visit:\n  ${url}`)
  openBrowser(url)
  consola.start('Waiting for you to finish signing in')

  const deadline = Date.now() + POLL_TIMEOUT_MS
  while (Date.now() < deadline) {
    await sleep(POLL_INTERVAL_MS)
    let result: (TokenPayload & { pending?: boolean }) | null
    try {
      result = await api<TokenPayload & { pending?: boolean }>('/api/plugin-auth/poll', { method: 'POST', query: { readKey } })
    } catch (error) {
      if (error instanceof ApiError && (error.status === 404 || error.status === 410)) {
        throw new Error('The sign-in session expired. Run the command again to start over.')
      }
      // Transient network / 5xx — keep polling
      continue
    }
    if (result && typeof result.access_token === 'string') {
      const credentials = fromTokens(result)
      await saveCredentials(credentials)
      return credentials
    }
  }

  throw new Error('Timed out waiting for sign-in. Run the command again to start over.')
}

async function refresh(credentials: Credentials): Promise<Credentials | null> {
  if (!credentials.refreshToken) return null
  try {
    const tokens = await api<TokenPayload>('/api/plugin-auth/refresh', {
      method: 'POST',
      body: { refresh_token: credentials.refreshToken }
    })
    const next = fromTokens({ ...tokens, refresh_token: tokens.refresh_token ?? credentials.refreshToken })
    await saveCredentials(next)
    return next
  } catch {
    return null
  }
}

export async function fetchMe(credentials: Credentials): Promise<Me> {
  return api<Me>('/api/plugin/me', { token: credentials.accessToken })
}

/**
 * Stored credentials + identity if they still verify (refreshing when
 * needed), or null. Never prompts — the preset funnel decides what to offer
 * a signed-out user instead of forcing a browser round-trip up front.
 */
export async function getSignedInState(): Promise<{ credentials: Credentials, me: Me } | null> {
  let credentials = await loadCredentials()
  if (!credentials) return null
  if (credentials.expiresAt && credentials.expiresAt - REFRESH_SKEW_MS < Date.now()) {
    credentials = await refresh(credentials)
    if (!credentials) return null
  }
  try {
    return { credentials, me: await fetchMe(credentials) }
  } catch (error) {
    if (!(error instanceof ApiError) || error.status !== 401) throw error
    const refreshed = await refresh(credentials)
    if (!refreshed) return null
    try {
      return { credentials: refreshed, me: await fetchMe(refreshed) }
    } catch (retryError) {
      if (retryError instanceof ApiError && retryError.status === 401) return null
      throw retryError
    }
  }
}

/**
 * Return working credentials: stored ones if they still verify (refreshing
 * when expired or rejected), otherwise a fresh interactive sign-in.
 */
export async function ensureSignedIn(): Promise<{ credentials: Credentials, me: Me }> {
  let credentials = await loadCredentials()

  if (credentials) {
    if (credentials.expiresAt && credentials.expiresAt - REFRESH_SKEW_MS < Date.now()) {
      credentials = await refresh(credentials)
    }
    if (credentials) {
      try {
        return { credentials, me: await fetchMe(credentials) }
      } catch (error) {
        if (!(error instanceof ApiError) || error.status !== 401) throw error
        const refreshed = await refresh(credentials)
        if (refreshed) {
          try {
            return { credentials: refreshed, me: await fetchMe(refreshed) }
          } catch (retryError) {
            if (!(retryError instanceof ApiError) || retryError.status !== 401) throw retryError
          }
        }
      }
    }
  }

  const fresh = await login()
  return { credentials: fresh, me: await fetchMe(fresh) }
}
