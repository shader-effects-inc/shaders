#!/usr/bin/env node
/**
 * Prepare a release of the `shaders` package.
 *
 *   pnpm release            → asks patch / minor / major
 *   pnpm release --minor    → no prompt
 *
 * From a clean, up-to-date `main` this creates `release/<version>`, writes the new
 * section of CHANGELOG.md from the commits since the last tag, sets the version in
 * packages/shaders/package.json, commits, pushes, and opens the release PR.
 *
 * Nothing is published here. Merging the PR is the release: the Release workflow sees a
 * version on `main` without a tag, verifies the changelog entry, builds, publishes to npm,
 * tags the merge commit and creates the GitHub Release from the same changelog section.
 */
import { execSync } from 'node:child_process'
import { existsSync, readFileSync, writeFileSync, mkdtempSync } from 'node:fs'
import { tmpdir } from 'node:os'
import { join } from 'node:path'
import { createInterface } from 'node:readline/promises'
import { generateMarkDown, getGitDiff, getLastGitTag, loadChangelogConfig, parseCommits } from 'changelogen'

const PACKAGE_JSON = 'packages/shaders/package.json'
const CHANGELOG = 'CHANGELOG.md'

const run = (cmd, options = {}) => execSync(cmd, { encoding: 'utf8', stdio: ['ignore', 'pipe', 'inherit'], ...options }).trim()
/** Like run, but a failing command yields null instead of throwing (for existence checks). */
const quiet = (cmd) => { try { return run(cmd, { stdio: ['ignore', 'pipe', 'ignore'] }) } catch { return null } }
const fail = (message) => { console.error(`\n✖ ${message}\n`); process.exit(1) }

function nextVersion(current, bump) {
  const [major, minor, patch] = current.split('.').map(Number)
  if (bump === 'major') return `${major + 1}.0.0`
  if (bump === 'minor') return `${major}.${minor + 1}.0`
  return `${major}.${minor}.${patch + 1}`
}

// ── 1. A clean, current main ──────────────────────────────────────────────
if (run('git status --porcelain')) fail('The working tree has uncommitted changes. Commit or stash them first.')
if (!quiet('gh --version')) fail('The GitHub CLI (gh) is needed to open the release PR: https://cli.github.com')
run('git checkout main')
run('git pull --ff-only')

const pkg = JSON.parse(readFileSync(PACKAGE_JSON, 'utf8'))
const lastTag = await getLastGitTag()
if (!lastTag) fail('No release tag found. Tag the last published version first (e.g. v3.2.475).')
if (!quiet(`git rev-parse -q --verify refs/tags/v${pkg.version}`)) {
  fail(`package.json is at ${pkg.version} but that version has no tag; a release may already be in flight.`)
}

// ── 2. Which bump ─────────────────────────────────────────────────────────
let bump = process.argv.find((a) => ['--patch', '--minor', '--major'].includes(a))?.slice(2)
if (!bump) {
  const rl = createInterface({ input: process.stdin, output: process.stdout })
  const answer = (await rl.question(`Current version is ${pkg.version} (last tag ${lastTag}). Bump patch, minor or major? [patch] `)).trim().toLowerCase()
  rl.close()
  bump = answer || 'patch'
}
if (!['patch', 'minor', 'major'].includes(bump)) fail(`Unknown bump "${bump}". Use patch, minor or major.`)
const version = nextVersion(pkg.version, bump)
const branch = `release/${version}`

// ── 3. The changelog section ──────────────────────────────────────────────
const config = await loadChangelogConfig(process.cwd(), { from: lastTag, to: 'HEAD', newVersion: version })
const commits = parseCommits(await getGitDiff(config.from, config.to), config)
// Only commits of a configured type (feat, fix, …) are listed; anything else is invisible
const listed = commits.filter((commit) => commit.type in config.types)
let section = (await generateMarkDown(commits, config)).trim().replace(/^### ❤️ Contributors$/m, '### Contributors')
const commitCount = Number(run(`git rev-list --count ${lastTag}..HEAD`))
if (listed.length === 0) {
  section = section.replace(/\n\n### Contributors[\s\S]*$/, '')
  section += `\n\n_${commitCount} commit${commitCount === 1 ? '' : 's'} since ${lastTag}, none with a conventional type (feat, fix, docs, …). Describe the changes here before merging._`
}

const existing = existsSync(CHANGELOG) ? readFileSync(CHANGELOG, 'utf8') : '# Changelog\n'
const firstEntry = existing.search(/^## /m)
const changelog = firstEntry === -1
  ? `${existing.trimEnd()}\n\n${section}\n`
  : `${existing.slice(0, firstEntry).trimEnd()}\n\n${section}\n\n${existing.slice(firstEntry)}`

// ── 4. Branch, files, commit, push, PR ────────────────────────────────────
// Anything failing from here on puts the checkout back on a clean `main` and removes the
// release branch this run created (remotely too, if the push had gone through), so a retry
// starts fresh. A branch that already existed before this run is left alone.
let created = false
let pushed = false
function abandon(error) {
  console.error(`\n✖ ${error?.message ?? error}`)
  console.error(created ? `  Cleaning up: back to main, removing ${branch}.` : '  Cleaning up: back to main.')
  quiet('git checkout -f main')
  if (created) quiet(`git branch -D ${branch}`)
  if (pushed) quiet(`git push origin --delete ${branch}`)
  process.exit(1)
}

let url
try {
  run(`git checkout -b ${branch}`)
  created = true
  writeFileSync(CHANGELOG, changelog)
  pkg.version = version
  writeFileSync(PACKAGE_JSON, `${JSON.stringify(pkg, null, 2)}\n`)
  run(`git add ${CHANGELOG} ${PACKAGE_JSON}`)
  run(`git commit -m "chore(release): v${version}"`)
  run(`git push -u origin ${branch}`)
  pushed = true
  url = openPullRequest()
} catch (error) {
  abandon(error)
}

function openPullRequest() {
const body = [
  `Release **v${version}** of \`shaders\`. Merging this PR publishes to npm, tags the merge commit and creates the GitHub Release.`,
  '',
  'Review the changelog below; edit `CHANGELOG.md` on this branch if the wording needs work.',
  '',
  '---',
  '',
  section,
].join('\n')
const bodyFile = join(mkdtempSync(join(tmpdir(), 'shaders-release-')), 'body.md')
writeFileSync(bodyFile, body)
return run(`gh pr create --base main --head ${branch} --title "Release v${version}" --body-file "${bodyFile}"`)
}

console.log(`\n✔ Release PR opened: ${url}`)
console.log(`  ${listed.length} of ${commitCount} commit${commitCount === 1 ? '' : 's'} since ${lastTag} appear in the changelog. Merge the PR to publish v${version}.\n`)
