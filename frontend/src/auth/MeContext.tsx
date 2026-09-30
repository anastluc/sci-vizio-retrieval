import { type ReactNode, createContext, useCallback, useContext, useEffect, useState } from "react";

import { api } from "../api/client";
import type { Me } from "../api/types";

interface MeState {
  me: Me | null;
  error: string | null;
  refresh: () => Promise<Me | null>;
}

const MeContext = createContext<MeState | null>(null);

/** The signed-in user's profile, plan and usage from the API. */
export function MeProvider({ children }: { children: ReactNode }) {
  const [me, setMe] = useState<Me | null>(null);
  const [error, setError] = useState<string | null>(null);

  const refresh = useCallback(async () => {
    try {
      const data = await api.me();
      setMe(data);
      setError(null);
      return data;
    } catch (e) {
      setError(e instanceof Error ? e.message : "Could not load your account");
      return null;
    }
  }, []);

  useEffect(() => {
    void refresh();
  }, [refresh]);

  return <MeContext.Provider value={{ me, error, refresh }}>{children}</MeContext.Provider>;
}

export function useMe(): MeState {
  const ctx = useContext(MeContext);
  if (!ctx) throw new Error("useMe must be used inside <MeProvider>");
  return ctx;
}
