// `npx shaders open`: open the connected Shaders project in the desktop app's editor,
// through its `shaders://` deep link. The editor is desktop-only; browsers get /download.
import { consola } from 'consola'
import { api, ApiError, getApiUrl, projectGone } from './api'
import { getSignedInState, openBrowser } from './auth'
import { resolveProjectContext } from './project'

/** The desktop app's deep-link scheme (ToDesktop "Support App Protocol"). */
const DESKTOP_PROTOCOL = 'shaders'

export async function open(): Promise<void> {
  const ctx = await resolveProjectContext(process.cwd())
  if (!ctx.project) {
    throw new Error('This codebase isn\'t connected to a Shaders project yet. Run npx shaders connect to set one up.')
  }

  // When we can (stored sign-in), make sure the project still exists so a
  // deleted one gets a clear message instead of an empty editor.
  // Signed out: just open the link; the app handles sign-in from there.
  try {
    const state = await getSignedInState()
    if (state) {
      const { projects } = await api<{ projects: { id: string }[] }>('/api/plugin/projects', {
        token: state.credentials.accessToken,
        query: { id: ctx.project, limit: '1' }
      })
      if (!projects.some(p => p.id === ctx.project)) throw projectGone(ctx.project)
    }
  } catch (error) {
    // API trouble (including while checking the sign-in itself) shouldn't
    // block opening a link we already know. A deleted project is a GoneError,
    // not an ApiError, so it still surfaces.
    if (!(error instanceof ApiError)) throw error
  }

  // The OS hands `shaders://` to the installed app, launching it if needed. We
  // can't tell from here whether it's installed, so say where to get it.
  consola.info(`Opening your project in the Shaders app…\n  Don't have it? Download it at ${getApiUrl()}/download`)
  openBrowser(`${DESKTOP_PROTOCOL}://desktop/open?project=${encodeURIComponent(ctx.project)}`)
}
