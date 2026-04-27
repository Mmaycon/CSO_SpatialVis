import { useEffect, useMemo, useRef } from "react";

/**
 * Returns a **stable** function reference so parent `useCallback`/`useEffect` deps
 * are not invalidated every render (which would re-run effects and **reset** debounce
 * timers on every re-render — e.g. viewport /cells never refetches after zoom).
 */
export function useDebouncedCallback<T extends (...args: Parameters<T>) => void>(
  fn: T,
  delayMs: number,
): T {
  const t = useRef<ReturnType<typeof setTimeout> | null>(null);
  const fnRef = useRef(fn);
  fnRef.current = fn;
  const delayRef = useRef(delayMs);
  delayRef.current = delayMs;

  useEffect(() => {
    return () => {
      if (t.current) clearTimeout(t.current);
    };
  }, []);

  return useMemo(() => {
    return ((...args: Parameters<T>) => {
      if (t.current) clearTimeout(t.current);
      t.current = setTimeout(() => {
        t.current = null;
        fnRef.current(...args);
      }, delayRef.current);
    }) as T;
  }, []) as T;
}
