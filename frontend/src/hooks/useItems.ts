import { useCallback, useEffect, useState } from "react";
import { api } from "../api/client";
import type { ItemSummary } from "../api/types";
import { messageFromError } from "../lib/format";

interface UseItemsResult {
  items: ItemSummary[];
  loading: boolean;
  error: string | null;
  refresh: () => Promise<void>;
}

/** Loads the saved-items list and exposes a refresh() for after an ingest. */
export function useItems(): UseItemsResult {
  const [items, setItems] = useState<ItemSummary[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  const refresh = useCallback(async () => {
    setLoading(true);
    setError(null);
    try {
      const data = await api.listItems();
      setItems(data.items);
    } catch (err) {
      setError(messageFromError(err));
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    void refresh();
  }, [refresh]);

  return { items, loading, error, refresh };
}
