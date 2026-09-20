import { AskPanel } from "./components/AskPanel";
import { IngestForm } from "./components/IngestForm";
import { ItemsList } from "./components/ItemsList";
import { Card } from "./components/ui";
import { useItems } from "./hooks/useItems";

export default function App() {
  const { items, loading, error, refresh } = useItems();

  return (
    <div className="min-h-screen bg-slate-100 text-slate-900">
      <header className="border-b border-slate-200 bg-white">
        <div className="mx-auto max-w-5xl px-6 py-5">
          <h1 className="text-xl font-bold tracking-tight">AI Knowledge Inbox</h1>
          <p className="mt-1 text-sm text-slate-500">
            Save notes and URLs, then ask questions answered from your own content — with cited
            sources.
          </p>
        </div>
      </header>

      <main className="mx-auto grid max-w-5xl grid-cols-1 gap-6 px-6 py-8 lg:grid-cols-2">
        <div className="space-y-6">
          <Card title="Add to inbox">
            <IngestForm onIngested={refresh} />
          </Card>
          <Card title="Saved items">
            <ItemsList items={items} loading={loading} error={error} onRefresh={refresh} />
          </Card>
        </div>

        <div>
          <Card title="Ask">
            <AskPanel hasItems={items.length > 0} />
          </Card>
        </div>
      </main>

      <footer className="mx-auto max-w-5xl px-6 pb-10 text-xs text-slate-400">
        Storage is in-memory on the backend — restarting the server clears everything.
      </footer>
    </div>
  );
}
