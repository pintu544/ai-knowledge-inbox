import { useState } from "react";
import { api } from "../api/client";
import type { QueryResponse } from "../api/types";
import { messageFromError } from "../lib/format";
import { AnswerView } from "./AnswerView";
import { ErrorBanner, Spinner } from "./ui";

interface AskPanelProps {
  /** Whether anything has been ingested; disables asking when empty. */
  hasItems: boolean;
}

export function AskPanel({ hasItems }: AskPanelProps) {
  const [question, setQuestion] = useState("");
  const [asking, setAsking] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [result, setResult] = useState<QueryResponse | null>(null);
  const [focusedCitation, setFocusedCitation] = useState<number | null>(null);

  async function handleSubmit(event: React.FormEvent) {
    event.preventDefault();
    if (!question.trim() || asking) return;

    setAsking(true);
    setError(null);
    setFocusedCitation(null);
    try {
      const response = await api.query({ question: question.trim() });
      setResult(response);
    } catch (err) {
      setError(messageFromError(err));
    } finally {
      setAsking(false);
    }
  }

  function focusCitation(citation: number) {
    setFocusedCitation(citation);
    document
      .getElementById(`source-${citation}`)
      ?.scrollIntoView({ behavior: "smooth", block: "center" });
  }

  return (
    <div className="space-y-5">
      <form onSubmit={handleSubmit} className="space-y-3">
        <label htmlFor="question" className="block text-sm font-medium text-slate-700">
          Ask a question about your saved content
        </label>
        <textarea
          id="question"
          value={question}
          rows={2}
          maxLength={2000}
          onChange={(e) => setQuestion(e.target.value)}
          onKeyDown={(e) => {
            if (e.key === "Enter" && (e.metaKey || e.ctrlKey)) handleSubmit(e);
          }}
          placeholder="e.g. What did I save about advisory locks?"
          className="w-full resize-y rounded-lg border border-slate-300 px-3 py-2 text-sm outline-none focus:border-slate-500 focus:ring-2 focus:ring-slate-200"
        />
        <div className="flex items-center gap-3">
          <button
            type="submit"
            disabled={asking || !question.trim() || !hasItems}
            className="inline-flex items-center gap-2 rounded-lg bg-slate-900 px-4 py-2 text-sm font-semibold text-white transition hover:bg-slate-700 disabled:cursor-not-allowed disabled:opacity-50"
          >
            {asking ? "Thinking…" : "Ask"}
          </button>
          {!hasItems && (
            <span className="text-sm text-slate-400">Save something first to ask a question.</span>
          )}
          <span className="ml-auto text-xs text-slate-400">⌘/Ctrl + Enter</span>
        </div>
      </form>

      {error && <ErrorBanner message={error} />}
      {asking && <Spinner label="Retrieving context and generating an answer…" />}
      {result && !asking && (
        <AnswerView
          result={result}
          onCitationFocus={focusCitation}
          focusedCitation={focusedCitation}
        />
      )}
    </div>
  );
}
