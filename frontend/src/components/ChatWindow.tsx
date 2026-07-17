import { useEffect, useRef, useState } from "react";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";
import { type ChatMessageOut, listMessages, streamChat } from "../api/client";

type PendingMessage = {
  text: string;
  status: "sending" | "error";
  error?: string;
};

function formatTimestamp(iso: string): string {
  return new Date(iso).toLocaleTimeString([], { hour: "numeric", minute: "2-digit" });
}

export function ChatWindow() {
  const [sessionId, setSessionId] = useState<string | null>(null);
  const [messages, setMessages] = useState<ChatMessageOut[]>([]);
  const [draft, setDraft] = useState("");
  const [streamingText, setStreamingText] = useState("");
  const [busy, setBusy] = useState(false);
  const [pendingMessage, setPendingMessage] = useState<PendingMessage | null>(null);
  const [liveStatus, setLiveStatus] = useState("");
  const requestIdRef = useRef(0);
  const messagesEndRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ block: "end" });
  }, [messages, streamingText, pendingMessage]);

  async function send(message: string) {
    if (!message || busy) return;
    const myRequestId = ++requestIdRef.current;
    setBusy(true);
    setDraft("");
    setStreamingText("");
    setPendingMessage({ text: message, status: "sending" });
    setLiveStatus("Assistant is responding…");

    let currentSessionId = sessionId;
    try {
      await streamChat(sessionId, message, {
        onSession: (id) => {
          if (myRequestId !== requestIdRef.current) return;
          currentSessionId = id;
          setSessionId(id);
        },
        onToken: (token) => {
          if (myRequestId !== requestIdRef.current) return;
          setStreamingText((prev) => prev + token);
        },
        onDone: () => {},
      });
      if (myRequestId !== requestIdRef.current) return;
      if (currentSessionId) {
        const fresh = await listMessages(currentSessionId);
        if (myRequestId !== requestIdRef.current) return;
        setMessages(fresh);
        setPendingMessage(null);
      }
      setLiveStatus("Response ready");
    } catch (err) {
      if (myRequestId !== requestIdRef.current) return;
      setPendingMessage((prev) => (prev ? { ...prev, status: "error", error: (err as Error).message } : prev));
      setLiveStatus("Response failed");
    } finally {
      if (myRequestId === requestIdRef.current) {
        setStreamingText("");
        setBusy(false);
      }
    }
  }

  function handleSend() {
    send(draft.trim());
  }

  function handleRetry() {
    if (pendingMessage) send(pendingMessage.text);
  }

  function handleNewChat() {
    if (busy && !window.confirm("A response is still in progress. Start a new chat anyway?")) return;
    requestIdRef.current++;
    setSessionId(null);
    setMessages([]);
    setPendingMessage(null);
    setStreamingText("");
    setDraft("");
    setBusy(false);
    setLiveStatus("");
  }

  const isEmpty = messages.length === 0 && !pendingMessage && !busy;

  return (
    <div className="panel chat-panel">
      <div className="panel-header">
        <h2>Chat</h2>
        <button className="btn btn-ghost" onClick={handleNewChat}>
          New chat
        </button>
      </div>

      {isEmpty ? (
        <div className="chat-empty-state">Ask a question about your uploaded documents to get started.</div>
      ) : (
        <div className="chat-messages">
          {messages.map((m) => (
            <div key={m.id} className={`chat-message chat-${m.role}`}>
              <div className="chat-message-header">
                <span className="chat-role">{m.role}</span>
                <span className="chat-timestamp">{formatTimestamp(m.created_at)}</span>
              </div>
              <div className="chat-content">
                {m.role === "assistant" ? (
                  <ReactMarkdown
                    remarkPlugins={[remarkGfm]}
                    components={{
                      a: ({ href, children }) => (
                        <a href={href} target="_blank" rel="noopener noreferrer">
                          {children}
                        </a>
                      ),
                    }}
                  >
                    {m.content}
                  </ReactMarkdown>
                ) : (
                  m.content
                )}
              </div>
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
          {pendingMessage && (
            <div className={`chat-message chat-user ${pendingMessage.status === "error" ? "chat-pending-error" : "chat-pending"}`}>
              <div className="chat-message-header">
                <span className="chat-role">user</span>
              </div>
              <div className="chat-content">{pendingMessage.text}</div>
              {pendingMessage.status === "error" && (
                <>
                  <div className="chat-error-text" role="alert">
                    {pendingMessage.error}
                  </div>
                  <button className="btn btn-ghost chat-retry-btn" onClick={handleRetry}>
                    Retry
                  </button>
                </>
              )}
            </div>
          )}
          {busy && (
            <div className="chat-message chat-assistant">
              <div className="chat-message-header">
                <span className="chat-role">assistant</span>
              </div>
              <div className="chat-content">{streamingText || "…"}</div>
            </div>
          )}
          <div ref={messagesEndRef} />
        </div>
      )}

      <span className="sr-only" aria-live="polite">
        {liveStatus}
      </span>

      <div className="chat-input-row">
        <label htmlFor="chat-draft-input" className="sr-only">
          Ask a question about your documents
        </label>
        <input
          id="chat-draft-input"
          value={draft}
          onChange={(e) => setDraft(e.target.value)}
          onKeyDown={(e) => e.key === "Enter" && handleSend()}
          placeholder="Ask a question about your documents..."
          disabled={busy}
        />
        <button className="btn btn-primary" onClick={handleSend} disabled={busy} aria-busy={busy}>
          {busy && <span className="spinner" aria-hidden="true" />}
          Send
        </button>
      </div>
    </div>
  );
}
