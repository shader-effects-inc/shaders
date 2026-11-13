// `npx shaders search <query>` — natural-language search over the curated
// preset library (the same index as the MCP search-presets tool). Public, no
// sign-in needed. The wizard reuses `searchPresets` + `printResults` and adds a
// picker underneath. Presets have no pages on shaders.com (they live in the
// desktop app's Pro library), so results carry no links.
import { consola } from 'consola'
import { api } from './api'

export interface SearchFlags {
  json: boolean
  limit?: number
}

export interface SearchResult {
  id: string
  slug: string | null
  title: string | null
  collection: { slug: string | null, name: string | null }
  similarity: number | null
  thumbnail: string | null
  description: string | null
  colors: string[]
}

export interface SearchResponse {
  query: string
  fallback: boolean
  results: SearchResult[]
}

const DESCRIPTION_CHARS = 110

function oneLine(text: string | null): string {
  if (!text) return ''
  const flat = text.replace(/\s+/g, ' ').trim()
  return flat.length > DESCRIPTION_CHARS ? `${flat.slice(0, DESCRIPTION_CHARS - 1).trimEnd()}…` : flat
}

export async function searchPresets(query: string, limit?: number): Promise<SearchResponse> {
  const queryParams: Record<string, string> = { q: query }
  if (limit) queryParams.limit = String(limit)
  return api<SearchResponse>('/api/plugin/presets/search', { query: queryParams })
}

/** One line per result: name, match, description. */
export function printResults(results: SearchResult[]): void {
  const width = Math.max(...results.map(r => (r.slug ?? r.id).length))
  consola.log('')
  for (const preset of results) {
    const name = (preset.slug ?? preset.id).padEnd(width)
    const match = preset.similarity != null ? `${Math.round(preset.similarity * 100)}%`.padStart(4) : '    '
    const description = oneLine(preset.description)
    consola.log(`  ${name}  ${match}  ${description}`)
  }
  consola.log('')
}

export async function search(args: string[], flags: SearchFlags): Promise<void> {
  const query = args.join(' ').trim()
  if (!query) throw new Error('What are you looking for? e.g. npx shaders search "liquid chrome hero background"')

  const { results, fallback } = await searchPresets(query, flags.limit)

  if (flags.json) {
    console.log(JSON.stringify({ query, fallback, results }, null, 2))
    return
  }

  if (!results.length) {
    consola.info(`No presets match "${query}". Try describing the look differently.`)
    return
  }

  printResults(results)
  consola.log(`Install one:   npx shaders install <name>   (presets require Shaders Pro)`)
}
