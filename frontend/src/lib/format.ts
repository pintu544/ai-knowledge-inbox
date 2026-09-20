export function formatTimestamp(iso: string): string {
  const date = new Date(iso);
  if (Number.isNaN(date.getTime())) return iso;
  return date.toLocaleString(undefined, {
    dateStyle: "medium",
    timeStyle: "short",
  });
}

export function messageFromError(error: unknown): string {
  if (error instanceof Error) return error.message;
  return "Something went wrong.";
}
