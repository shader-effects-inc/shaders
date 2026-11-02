// `npx shaders update [<id>...]` — refresh installed shaders from shaders.com.
// With no IDs, every entry in shaders.lock.json is checked. Files with local
// edits are left alone unless --force.
import { consola } from 'consola'
import { GoneError } from './api'
import { ensureSignedIn } from './auth'
import { componentLines, fetchCode, readIfExists, resolveInsideProject, upgradeToClone, writeShaderFile, type InstallFlags } from './install'
import { ensureProject, pricingUrl, resolvePreset } from './presets'
import { hashContent, readLockFile, writeLockFile } from './lockFile'
import { resolveProjectContext } from './project'

export async function update(ids: string[], flags: InstallFlags): Promise<void> {
  const ctx = await resolveProjectContext(process.cwd())
  const lock = await readLockFile(ctx.detected.dir)

  const targets = ids.length ? ids : Object.keys(lock.shaders)
  if (targets.length === 0) {
    consola.info('Nothing installed yet — run npx shaders install <id> first')
    return
  }
  for (const id of targets) {
    if (!lock.shaders[id]) throw new Error(`${id} isn't in shaders.lock.json — install it first with npx shaders install ${id}`)
  }

  const { credentials, me } = await ensureSignedIn()

  const counts = { updated: 0, unchanged: 0, skipped: 0, gone: 0, interrupted: 0, failed: 0 }
  for (const [index, id] of targets.entries()) {
    const existing = lock.shaders[id]
    try {
      // A clone that was created but whose file never got written (crash /
      // network blip between the two phases): finish the job. Goes before the
      // local-edit guard because a pending entry may have no file of ours yet
      // (hash '') — writeComponentFile applies the right check and message.
      if (existing.pending && existing.source) {
        const preset = await resolvePreset(existing.source.presetId, flags)
        const projectId = await ensureProject(ctx, credentials, flags)
        await upgradeToClone(ctx, credentials, preset, projectId, existing, id, lock, flags.force)
        counts.updated++
        continue
      }

      // Refuse to silently clobber hand-edited files (unless --force). Check
      // before the network round-trip so the message is immediate.
      const current = await readIfExists(resolveInsideProject(ctx.detected.dir, existing.file))
      if (current !== null && hashContent(current) !== existing.hash && !flags.force) {
        counts.skipped++
        consola.warn(`Skipped "${existing.title}" — ${existing.file} has local edits (use --force to overwrite)`)
        continue
      }

      // Watermarked preview placeholder: with Pro, clone the preset into the
      // project and replace the file with real source (re-keys the entry).
      if (existing.source?.kind === 'preview') {
        if (!me.isPro) {
          counts.skipped++
          consola.info(`"${existing.title}" is a watermarked preview — unlock Shaders Pro to swap in the real component: ${pricingUrl(existing.source.slug)}`)
          continue
        }
        const preset = await resolvePreset(existing.source.presetId, flags)
        const projectId = await ensureProject(ctx, credentials, flags)
        await upgradeToClone(ctx, credentials, preset, projectId, existing, id, lock, flags.force)
        counts.updated++
        continue
      }

      const response = await fetchCode(credentials, id, existing.framework)
      const { entry, status, components } = await writeShaderFile(ctx, response, existing, lock, flags.force)
      lock.shaders[id] = entry
      await writeLockFile(ctx.detected.dir, lock)

      if (status === 'unchanged' && components.length === 0) {
        counts.unchanged++
        consola.log(`  "${entry.title}" is up to date`)
      } else {
        counts.updated++
        consola.success(`${current === null ? 'Restored' : 'Updated'} "${entry.title}" → ${entry.file}${componentLines(components)}`)
      }
    } catch (error) {
      if (error instanceof GoneError) {
        // Nothing broke locally — the source just isn't there anymore.
        counts.gone++
        consola.warn(`${existing.title} (${id}): ${error.message}\n  ${existing.file} was left untouched.`)
        if (error.what === 'project') {
          // Every remaining entry would hit the same wall — stop, but say so.
          counts.interrupted = targets.length - index - 1
          if (counts.interrupted) consola.warn(`Stopped — ${counts.interrupted} more shader${counts.interrupted === 1 ? '' : 's'} not checked`)
          break
        }
        continue
      }
      counts.failed++
      consola.error(`${existing.title} (${id}): ${error instanceof Error ? error.message : String(error)}`)
    }
  }

  const parts = [
    counts.updated && `${counts.updated} updated`,
    counts.unchanged && `${counts.unchanged} up to date`,
    counts.skipped && `${counts.skipped} skipped`,
    counts.gone && `${counts.gone} unavailable in Shaders`,
    counts.interrupted && `${counts.interrupted} not checked`,
    counts.failed && `${counts.failed} failed`
  ].filter(Boolean)
  consola.log(`\n${parts.join(', ')}`)
  // Anything that didn't get a definite answer is a non-zero exit, so CI can tell.
  if (counts.failed || counts.interrupted) process.exit(1)
}
