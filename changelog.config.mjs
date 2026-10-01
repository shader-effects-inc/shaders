// Changelog generation (changelogen), used by `pnpm release` (scripts/release.mjs).
// Only these commit types appear in CHANGELOG.md; chores, CI and test commits stay out.
// The headings are plain words: the changelog is rendered on shaders.com as well as here.
export default {
  output: 'CHANGELOG.md',
  types: {
    feat: { title: 'Features', semver: 'minor' },
    fix: { title: 'Fixes', semver: 'patch' },
    perf: { title: 'Performance', semver: 'patch' },
    refactor: { title: 'Refactors', semver: 'patch' },
    docs: { title: 'Documentation', semver: 'patch' },
    types: { title: 'Types', semver: 'patch' },
  },
  templates: {
    tagBody: 'v{{newVersion}}',
  },
  // Contributors are credited by name (GitHub handle when resolvable), never by email
  hideAuthorEmail: true,
}
