import type { ItemSummary } from "../api/types";
import { formatTimestamp } from "../lib/format";
import { Badge, ErrorBanner, Spinner } from "./ui";

interface ItemsListProps {
  items: ItemSummary[];
  loading: boolean;
  error: string | null;
  onRefresh: () => void;
}

export function ItemsList({ items, loading, error, onRefresh }: ItemsListProps) {
  return (
    <div className="space-y-4">
      <div className="flex items-center justify-between">
        <span className="text-sm text-slate-500">
          {items.length} item{items.length === 1 ? "" : "s"} saved
        </span>
        <button
          type="button"
          onClick={onRefresh}
          className="text-sm font-medium text-slate-600 underline-offset-2 hover:underline"
        >
          Refresh
        </button>
      </div>

      {error && <ErrorBanner message={error} />}

      {loading && items.length === 0 ? (
        <Spinner label="Loading saved items…" />
      ) : items.length === 0 ? (
        <p className="rounded-lg border border-dashed border-slate-200 px-4 py-8 text-center text-sm text-slate-400">
          Nothing saved yet. Add a note or URL to get started.
        </p>
      ) : (
        <ul className="space-y-3">
          {items.map((item) => (
            <li key={item.id} className="rounded-lg border border-slate-200 p-4">
              <div className="mb-1 flex items-start justify-between gap-3">
                <h3 className="font-medium text-slate-900">{item.title}</h3>
                <Badge>{item.source_type}</Badge>
              </div>
              <p className="text-sm leading-relaxed text-slate-600">{item.preview}</p>
              <div className="mt-2 flex flex-wrap items-center gap-x-3 gap-y-1 text-xs text-slate-400">
                <span>{formatTimestamp(item.created_at)}</span>
                <span>·</span>
                <span>
                  {item.chunk_count} chunk{item.chunk_count === 1 ? "" : "s"}
                </span>
                <span>·</span>
                <span>{item.char_count.toLocaleString()} chars</span>
                {item.source_url && (
                  <>
                    <span>·</span>
                    <a
                      href={item.source_url}
                      target="_blank"
                      rel="noreferrer"
                      className="text-slate-500 underline-offset-2 hover:underline"
                    >
                      source
                    </a>
                  </>
                )}
              </div>
            </li>
          ))}
        </ul>
      )}
    </div>
  );
}
