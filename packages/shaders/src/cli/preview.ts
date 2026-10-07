// `npx shaders preview <name>` — open a preset's page on shaders.com so the
// user can see it running before `npx shaders install <name>`. Public, no
// sign-in: the page is the same one the library links to.
import { consola } from 'consola'
import { getApiUrl } from './api'
import { openBrowser } from './auth'
import { resolvePreset, type PresetSummary } from './presets'

/** Short link to the preset's page (shaders.com/p/<name>), the same name `install` takes. */
export function presetPageUrl(preset: Pick<PresetSummary, 'slug'>): string {
  const url = new URL(`/p/${encodeURIComponent(preset.slug)}`, getApiUrl() + '/')
  url.searchParams.set('utm_source', 'cli')
  url.searchParams.set('utm_medium', 'terminal')
  return url.toString()
}

export async function preview(args: string[], flags: { yes: boolean }): Promise<void> {
  const ref = args[0]
  if (!ref) throw new Error('Which preset? e.g. npx shaders preview offsets-1 (find names with npx shaders search)')
  if (args.length > 1) throw new Error(`Unexpected argument: ${args[1]} — preview takes one preset name`)

  const preset = await resolvePreset(ref, flags)
  const url = presetPageUrl(preset)
  consola.info(`Opening ${preset.title} — ${url}`)
  openBrowser(url)
  consola.log(`Install it with: npx shaders install ${preset.slug}`)
}
