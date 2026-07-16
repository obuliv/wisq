import { useState } from "react";
import { type ChatMessageOut, listMessages, streamChat } from "../api/client";

export function ChatWindow() {
  const [sessionId, setSessionId] = useState<string | null>(null);
  const [messages, setMessages] = useState<ChatMessageOut[]>([]);
  const [draft, setDraft] = useState("");
  const [streamingText, setStreamingText] = useState("");
  const [busy, setBusy] = useState(false);

  async function handleSend() {
    const message = draft.trim();
    if (!message || busy) return;
    setBusy(true);
    setDraft("");
    setStreamingText("");

    let currentSessionId = sessionId;
    try {
      await streamChat(sessionId, message, {
        onSession: (id) => {
          currentSessionId = id;
          setSessionId(id);
        },
        onToken: (token) => setStreamingText((prev) => prev + token),
        onDone: () => {},
      });
      if (currentSessionId) {
        setMessages(await listMessages(currentSessionId));
      }
    } catch (err) {
      setStreamingText(`Error: ${(err as Error).message}`);
    } finally {
      setStreamingText("");
      setBusy(false);
    }
  }

  return (
    <div className="panel chat-panel">
      <h2>Chat</h2>
      <div className="chat-messages">
        {messages.map((m) => (
          <div key={m.id} className={`chat-message chat-${m.role}`}>
            <div className="chat-role">{m.role}</div>
            <div className="chat-content">{m.content}</div>
            {m.sources && m.sources.length > 0 && (
              <details className="chat-sources">
                <summary>{m.sources.length} source(s)</summary>
                <ul>
                  {m.sources.map((s, i) => (
                    <li key={i}>
                      <em>{s.document_title ?? s.doc_id}</em>
                      {s.locator ? ` (${s.locator})` : ""}: {s.text}
                    </li>
                  ))}
                </ul>
              </details>
            )}
          </div>
        ))}
        {busy && (
          <div className="chat-message chat-assistant">
            <div className="chat-role">assistant</div>
            <div className="chat-content">{streamingText || "..."}</div>
          </div>
        )}
      </div>
      <div className="chat-input-row">
        <input
          value={draft}
          onChange={(e) => setDraft(e.target.value)}
          onKeyDown={(e) => e.key === "Enter" && handleSend()}
          placeholder="Ask a question about your documents..."
          disabled={busy}
        />
        <button onClick={handleSend} disabled={busy}>
          Send
        </button>
      </div>
    </div>
  );
}
