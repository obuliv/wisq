export interface DocumentOut {
  id: string;
  filename: string;
  content_type: string;
  status: "queued" | "processing" | "ready" | "failed";
  error_message: string | null;
  doc_type: string | null;
  title: string | null;
  version: string | null;
  effective_date: string | null;
  applicable_regions: { included: string[]; excluded: string[] } | null;
  applicable_personnel: { included: string[]; excluded: string[] } | null;
  is_latest: boolean;
  doc_metadata: Record<string, unknown> | null;
  created_at: string;
  updated_at: string;
}

export interface ChatMessageOut {
  id: string;
  role: "user" | "assistant";
  content: string;
  sources: Array<{
    doc_id: string;
    document_title: string | null;
    locator: string | null;
    text: string;
    score: number;
  }> | null;
  created_at: string;
}

export async function uploadDocument(file: File): Promise<DocumentOut> {
  const formData = new FormData();
  formData.append("file", file);
  const response = await fetch("/api/documents", { method: "POST", body: formData });
  if (!response.ok) throw new Error(await response.text());
  return response.json();
}

export async function listDocuments(): Promise<DocumentOut[]> {
  const response = await fetch("/api/documents");
  if (!response.ok) throw new Error(await response.text());
  return response.json();
}

export async function listMessages(sessionId: string): Promise<ChatMessageOut[]> {
  const response = await fetch(`/api/chat/sessions/${sessionId}/messages`);
  if (!response.ok) throw new Error(await response.text());
  return response.json();
}

export interface ChatStreamHandlers {
  onSession: (sessionId: string) => void;
  onToken: (token: string) => void;
  onDone: () => void;
}

/** Parses the backend's `text/event-stream` response (fetch, not EventSource,
 * since EventSource can't send a POST body). */
export async function streamChat(
  sessionId: string | null,
  message: string,
  handlers: ChatStreamHandlers,
): Promise<void> {
  const response = await fetch("/api/chat", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ session_id: sessionId, message }),
  });
  if (!response.ok || !response.body) throw new Error(await response.text());

  const reader = response.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";

  for (;;) {
    const { value, done } = await reader.read();
    if (done) break;
    buffer += decoder.decode(value, { stream: true });

    let boundary = buffer.indexOf("\n\n");
    while (boundary !== -1) {
      const frame = buffer.slice(0, boundary);
      buffer = buffer.slice(boundary + 2);
      dispatchFrame(frame, handlers);
      boundary = buffer.indexOf("\n\n");
    }
  }
}

function dispatchFrame(frame: string, handlers: ChatStreamHandlers): void {
  let event = "message";
  let data = "";
  for (const line of frame.split("\n")) {
    if (line.startsWith("event:")) event = line.slice("event:".length).trim();
    else if (line.startsWith("data:")) data = line.slice("data:".length).trim().replace(/\\n/g, "\n");
  }
  if (event === "session") handlers.onSession(data);
  else if (event === "token") handlers.onToken(data);
  else if (event === "done") handlers.onDone();
}
