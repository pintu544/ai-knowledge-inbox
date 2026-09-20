import { Fragment, useMemo } from "react";
import type { QueryResponse, SourceSnippet } from "../api/types";
import { Badge } from "./ui";

interface AnswerViewProps {
  result: QueryResponse;
  /** Called when a [n] marker (or a source card) is activated. */
  onCitationFocus: (citation: number) => void;
  focusedCitation: number | null;
}

// Split an answer into text + citation tokens. "[1]" and "[1, 2]" both work:
// each number inside the brackets becomes its own clickable chip.
const CITATION_RE = /\[(\d+(?:\s*,\s*\d+)*)\]/g;

interface Token {
  key: string;
  text?: string;
  citation?: number;
}

function tokenize(answer: string, validCitations: Set<number>): Token[] {
  const tokens: Token[] = [];
  let lastIndex = 0;
  let match: RegExpExecArray | null;
  let counter = 0;

  CITATION_RE.lastIndex = 0;
  while ((match = CITATION_RE.exec(answer)) !== null) {
    if (match.index > lastIndex) {
      tokens.push({ key: `t${counter++}`, text: answer.slice(lastIndex, match.index) });
    }
    const numbers = match[1].split(",").map((n) => Number(n.trim()));
    // Only treat as a citation if every number maps to a real source.
    if (numbers.every((n) => validCitations.has(n))) {
      numbers.forEach((n) => tokens.push({ key: `c${counter++}`, citation: n }));
    } else {
      tokens.push({ key: `t${counter++}`, text: match[0] });
    }
    lastIndex = match.index + match[0].length;
  }
  if (lastIndex < answer.length) {
    tokens.push({ key: `t${counter++}`, text: answer.slice(lastIndex) });
  }
  return tokens;
}

export function AnswerView({ result, onCitationFocus, focusedCitation }: AnswerViewProps) {
  const validCitations = useMemo(
    () => new Set(result.sources.map((s) => s.citation)),
    [result.sources],
  );
  const tokens = useMemo(
    () => tokenize(result.answer, validCitations),
    [result.answer, validCitations],
  );

  return (
    <div className="space-y-5">
      <div className="rounded-lg bg-slate-50 p-4 text-[15px] leading-7 text-slate-800">
        {tokens.map((token) =>
          token.text !== undefined ? (
            <Fragment key={token.key}>{token.text}</Fragment>
          ) : (
            <button
              key={token.key}
              type="button"
              onClick={() => onCitationFocus(token.citation!)}
              className="mx-0.5 inline-flex h-5 min-w-5 items-center justify-center rounded bg-slate-900 px-1 align-middle text-xs font-semibold text-white transition hover:bg-slate-700"
              title={`Jump to source ${token.citation}`}
            >
              {token.citation}
            </button>
          ),
        )}
      </div>

      <div className="flex flex-wrap items-center gap-2 text-xs text-slate-400">
        <span>model: {result.model}</span>
        <span>·</span>
        <span>
          {result.retrieved_chunk_count} chunk
          {result.retrieved_chunk_count === 1 ? "" : "s"} retrieved
        </span>
      </div>

      <div>
        <h3 className="mb-2 text-sm font-semibold text-slate-500">Sources</h3>
        {result.sources.length === 0 ? (
          <p className="text-sm text-slate-400">No sources were cited.</p>
        ) : (
          <ul className="space-y-3">
            {result.sources.map((source) => (
              <SourceCard
                key={`${source.item_id}-${source.chunk_index}`}
                source={source}
                focused={focusedCitation === source.citation}
                onFocus={() => onCitationFocus(source.citation)}
              />
            ))}
          </ul>
        )}
      </div>
    </div>
  );
}

function SourceCard({
  source,
  focused,
  onFocus,
}: {
  source: SourceSnippet;
  focused: boolean;
  onFocus: () => void;
}) {
  return (
    <li
      id={`source-${source.citation}`}
      onClick={onFocus}
      className={`cursor-pointer rounded-lg border p-4 transition ${
        focused ? "border-slate-900 ring-2 ring-slate-200" : "border-slate-200 hover:border-slate-300"
      }`}
    >
      <div className="mb-1 flex items-start justify-between gap-3">
        <div className="flex items-center gap-2">
          <span className="inline-flex h-5 min-w-5 items-center justify-center rounded bg-slate-900 px-1 text-xs font-semibold text-white">
            {source.citation}
          </span>
          <span className="font-medium text-slate-900">{source.title}</span>
        </div>
        <Badge>{source.source_type}</Badge>
      </div>
      <p className="text-sm leading-relaxed text-slate-600">{source.snippet}</p>
      <div className="mt-2 flex flex-wrap items-center gap-x-3 text-xs text-slate-400">
        <span>chunk #{source.chunk_index}</span>
        <span>·</span>
        <span>score {source.score.toFixed(3)}</span>
        {source.source_url && (
          <>
            <span>·</span>
            <a
              href={source.source_url}
              target="_blank"
              rel="noreferrer"
              onClick={(e) => e.stopPropagation()}
              className="text-slate-500 underline-offset-2 hover:underline"
            >
              open source
            </a>
          </>
        )}
      </div>
    </li>
  );
}
