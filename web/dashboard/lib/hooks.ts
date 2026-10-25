"use client";
import { useCallback, useEffect, useState } from "react";
import { api } from "./api";

export function useApi<T>(path: string | null, deps: unknown[] = []) {
  const [data, setData] = useState<T | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(!!path);
  const load = useCallback(async () => {
    if (!path) return;
    setLoading(true);
    try { setData(await api<T>(path)); setError(null); } catch (e) { setError((e as Error).message); } finally { setLoading(false); }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [path, ...deps]);
  useEffect(() => { void load(); }, [load]);
  return { data, error, loading, reload: load, setData };
}

export function useMe() {
  const [me, setMe] = useState<{ tenant_id: string; role: string; user_id?: string | null } | null>(null);
  useEffect(() => { fetch("/api/session").then((r) => (r.ok ? r.json() : null)).then(setMe).catch(() => {}); }, []);
  return me;
}
export const canEdit = (role?: string) => role === "editor" || role === "admin";
