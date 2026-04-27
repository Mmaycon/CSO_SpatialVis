import { useEffect, useRef } from "react";

export function useDebouncedCallback<T extends (...args: Parameters<T>) => void>(
  fn: T,
  delayMs: number,
): T {
  const t = useRef<ReturnType<typeof setTimeout> | null>(null);
  const fnRef = useRef(fn);
  fnRef.current = fn;

  useEffect(() => {
    return () => {
      if (t.current) clearTimeout(t.current);
    };
  }, []);

  return ((...args: Parameters<T>) => {
    if (t.current) clearTimeout(t.current);
    t.current = setTimeout(() => fnRef.current(...args), delayMs);
  }) as T;
}
