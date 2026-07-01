import { useCallback, useEffect, useState } from 'react'
import { api } from '../api/client'
import type { ExtensionWithStatus } from '../types/api'

export interface UseExtensionsResult {
  extensions: ExtensionWithStatus[]
  loading: boolean
  error: string | null
  refresh: () => void
}

export function useExtensions(): UseExtensionsResult {
  const [extensions, setExtensions] = useState<ExtensionWithStatus[]>([])
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)

  const refresh = useCallback(() => {
    setLoading(true)
    Promise.all([api.listExtensions(), api.listInstalledExtensions()])
      .then(([catalogData, installedData]) => {
        const installedMap = new Map(
          installedData.extensions.map((e) => [e.id, e]),
        )
        const merged: ExtensionWithStatus[] = catalogData.extensions.map((ext) => ({
          ...ext,
          auto_installed: installedMap.get(ext.id)?.auto_installed ?? false,
        }))
        setExtensions(merged)
        setError(null)
      })
      .catch((err: unknown) => {
        const msg = err instanceof Error ? err.message : 'Failed to load extensions.'
        setError(msg)
      })
      .finally(() => setLoading(false))
  }, [])

  useEffect(() => {
    refresh()
  }, [refresh])

  return { extensions, loading, error, refresh }
}
