import { useState } from "react";
import { api, ApiError } from "../api/client";
import type { SourceType } from "../api/types";
import { messageFromError } from "../lib/format";
import { ErrorBanner } from "./ui";

interface IngestFormProps {
  /** Called after a successful ingest so the parent can refresh the list. */
  onIngested: () => void;
}

const TABS: { value: SourceType; label: string; placeholder: string }[] = [
  {
    value: "note",
    label: "Note",
    placeholder: "Paste or type a note to save…",
  },
  {
    value: "url",
    label: "URL",
    placeholder: "https://example.com/article",
  },
];

export function IngestForm({ onIngested }: IngestFormProps) {
  const [sourceType, setSourceType] = useState<SourceType>("note");
  const [content, setContent] = useState("");
  const [title, setTitle] = useState("");
  const [submitting, setSubmitting] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [success, setSuccess] = useState<string | null>(null);

  const active = TABS.find((tab) => tab.value === sourceType)!;

  async function handleSubmit(event: React.FormEvent) {
    event.preventDefault();
    if (!content.trim() || submitting) return;

    setSubmitting(true);
    setError(null);
    setSuccess(null);
    try {
      const result = await api.ingest({
        source_type: sourceType,
        content: content.trim(),
        title: title.trim() || null,
      });
      setSuccess(
        `Saved “${result.item.title}” (${result.chunk_count} chunk${
          result.chunk_count === 1 ? "" : "s"
        }).`,
      );
      setContent("");
      setTitle("");
      onIngested();
    } catch (err) {
      // Prefer field-level validation messages when the backend provides them.
      if (err instanceof ApiError && err.fields.length > 0) {
        setError(err.fields.map((f) => f.message).join(" "));
      } else {
        setError(messageFromError(err));
      }
    } finally {
      setSubmitting(false);
    }
  }

  return (
    <form onSubmit={handleSubmit} className="space-y-4">
      <div className="inline-flex rounded-lg border border-slate-200 p-0.5">
        {TABS.map((tab) => (
          <button
            key={tab.value}
            type="button"
            onClick={() => {
              setSourceType(tab.value);
              setError(null);
            }}
            className={`rounded-md px-4 py-1.5 text-sm font-medium transition ${
              sourceType === tab.value
                ? "bg-slate-900 text-white"
                : "text-slate-600 hover:bg-slate-100"
            }`}
            aria-pressed={sourceType === tab.value}
          >
            {tab.label}
          </button>
        ))}
      </div>

      <div>
        <label htmlFor="title" className="mb-1 block text-sm font-medium text-slate-700">
          Title <span className="font-normal text-slate-400">(optional)</span>
        </label>
        <input
          id="title"
          type="text"
          value={title}
          maxLength={300}
          onChange={(e) => setTitle(e.target.value)}
          placeholder={sourceType === "url" ? "Defaults to the page title" : "A short label"}
          className="w-full rounded-lg border border-slate-300 px-3 py-2 text-sm outline-none focus:border-slate-500 focus:ring-2 focus:ring-slate-200"
        />
      </div>

      <div>
        <label htmlFor="content" className="mb-1 block text-sm font-medium text-slate-700">
          {sourceType === "url" ? "URL" : "Note"}
        </label>
        {sourceType === "url" ? (
          <input
            id="content"
            type="url"
            value={content}
            onChange={(e) => setContent(e.target.value)}
            placeholder={active.placeholder}
            className="w-full rounded-lg border border-slate-300 px-3 py-2 text-sm outline-none focus:border-slate-500 focus:ring-2 focus:ring-slate-200"
          />
        ) : (
          <textarea
            id="content"
            value={content}
            rows={5}
            maxLength={100_000}
            onChange={(e) => setContent(e.target.value)}
            placeholder={active.placeholder}
            className="w-full resize-y rounded-lg border border-slate-300 px-3 py-2 text-sm outline-none focus:border-slate-500 focus:ring-2 focus:ring-slate-200"
          />
        )}
      </div>

      {error && <ErrorBanner message={error} />}
      {success && (
        <div className="rounded-lg border border-emerald-200 bg-emerald-50 px-4 py-3 text-sm text-emerald-700">
          {success}
        </div>
      )}

      <button
        type="submit"
        disabled={submitting || !content.trim()}
        className="inline-flex items-center gap-2 rounded-lg bg-slate-900 px-4 py-2 text-sm font-semibold text-white transition hover:bg-slate-700 disabled:cursor-not-allowed disabled:opacity-50"
      >
        {submitting ? "Saving…" : sourceType === "url" ? "Fetch & save" : "Save note"}
      </button>
    </form>
  );
}
