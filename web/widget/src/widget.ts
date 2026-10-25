import { VoxaCall, type CallSnapshot, type UiState } from "@voxa/client";

/** Embeddable widget: <script src=".../widget.js" data-key="wk_..."></script>  (Shadow DOM, no CSS leakage). */
const script = document.currentScript as HTMLScriptElement | null;
const cfg = {
  key: script?.dataset.key ?? "",
  api: (script?.dataset.api ?? (script ? new URL(script.src).origin + "/api" : "")).replace(/\/$/, ""),
  title: script?.dataset.title ?? "Talk to us",
  color: script?.dataset.color ?? "#4f46e5",
  position: script?.dataset.position === "left" ? "left" : "right",
  theme: script?.dataset.theme ?? "auto",
  prompt: script?.dataset.prompt ?? "Start a voice conversation with our assistant.",
};

const LABEL: Record<UiState, string> = {
  idle: "Ready", connecting: "Connecting…", listening: "Listening", thinking: "Thinking…", speaking: "Speaking",
  paused: "Paused", handoff: "Connecting you to a person…", reconnecting: "Reconnecting…", ended: "Call ended", error: "Something went wrong",
};

const css = `
:host{all:initial}
*{box-sizing:border-box;font-family:system-ui,-apple-system,Segoe UI,Roboto,sans-serif}
.root{--accent:${cfg.color};--bg:#fff;--fg:#111827;--muted:#6b7280;--line:#e5e7eb;--card:#f9fafb;position:fixed;bottom:20px;${cfg.position}:20px;z-index:2147483000;color:var(--fg)}
.root.dark{--bg:#111827;--fg:#f9fafb;--muted:#9ca3af;--line:#374151;--card:#1f2937}
@media (prefers-color-scheme:dark){.root.auto{--bg:#111827;--fg:#f9fafb;--muted:#9ca3af;--line:#374151;--card:#1f2937}}
.fab{width:56px;height:56px;border-radius:50%;border:0;background:var(--accent);color:#fff;cursor:pointer;box-shadow:0 6px 20px rgba(0,0,0,.25);display:grid;place-items:center}
.fab:focus-visible,button:focus-visible{outline:3px solid #93c5fd;outline-offset:2px}
.panel{position:absolute;bottom:68px;${cfg.position}:0;width:340px;max-width:calc(100vw - 32px);background:var(--bg);border:1px solid var(--line);border-radius:16px;box-shadow:0 12px 40px rgba(0,0,0,.28);overflow:hidden;display:none}
.panel.open{display:block}
header{display:flex;align-items:center;justify-content:space-between;padding:12px 16px;border-bottom:1px solid var(--line);font-weight:600}
.x{background:none;border:0;color:var(--muted);cursor:pointer;font-size:20px;line-height:1}
.stage{display:grid;place-items:center;padding:20px 16px 8px}
.orb{width:84px;height:84px;border-radius:50%;background:var(--accent);opacity:.95;transform:scale(calc(1 + var(--lvl,0)*.9));transition:transform .08s linear,opacity .2s;position:relative}
.orb::after{content:"";position:absolute;inset:-8px;border-radius:50%;border:2px solid var(--accent);opacity:0}
.s-listening .orb::after{animation:ring 1.8s ease-out infinite}
.s-thinking .orb{animation:think 1.2s ease-in-out infinite}
.s-paused .orb{opacity:.45}.s-reconnecting .orb,.s-connecting .orb{opacity:.5;animation:think 1.2s ease-in-out infinite}
.s-error .orb{background:#dc2626}.s-ended .orb{background:var(--muted)}
@keyframes ring{0%{transform:scale(.9);opacity:.6}100%{transform:scale(1.5);opacity:0}}
@keyframes think{0%,100%{transform:scale(.92)}50%{transform:scale(1.04)}}
@media (prefers-reduced-motion:reduce){.orb,.orb::after{animation:none!important;transition:none}}
.status{text-align:center;font-size:14px;color:var(--muted);padding-bottom:8px;min-height:22px}
.caps{margin:0 16px;height:140px;overflow-y:auto;background:var(--card);border-radius:10px;padding:10px;font-size:14px;line-height:1.4}
.cap{margin:0 0 8px}.cap b{font-size:11px;text-transform:uppercase;letter-spacing:.04em;color:var(--muted);display:block}
.cap.user{opacity:.85}.cap.partial{opacity:.55}
.row{display:flex;gap:8px;padding:12px 16px;justify-content:center;align-items:center}
.btn{border:1px solid var(--line);background:var(--bg);color:var(--fg);padding:10px 14px;border-radius:10px;cursor:pointer;font-size:14px}
.btn.primary{background:var(--accent);color:#fff;border-color:var(--accent);font-weight:600}.btn.danger{background:#dc2626;color:#fff;border-color:#dc2626}
.btn[aria-pressed=true]{background:var(--fg);color:var(--bg)}
.note{font-size:12px;color:var(--muted);text-align:center;padding:0 16px 12px}
.stars{display:flex;gap:4px;justify-content:center}.stars button{font-size:26px;background:none;border:0;cursor:pointer;color:#f59e0b;padding:2px}
.center{text-align:center;padding:16px}
.sr{position:absolute;width:1px;height:1px;overflow:hidden;clip:rect(0 0 0 0)}
`;

const mic = '<svg width="26" height="26" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" aria-hidden="true"><rect x="9" y="3" width="6" height="11" rx="3"/><path d="M5 11a7 7 0 0 0 14 0M12 18v3"/></svg>';

class Widget {
  private host = document.createElement("div");
  private sh = this.host.attachShadow({ mode: "open" });
  private call = new VoxaCall();
  private open = false;
  private ended = false;
  private rated = false;
  private snap: CallSnapshot = this.call.snapshot;
  private busy = false;
  private msg = "";
  private csatToken = "";
  private resumeTried = false;

  constructor() {
    this.call.subscribe((s) => {
      this.snap = s;
      if (s.state === "error" && this.csatToken && !this.resumeTried && /connection lost/i.test(s.error ?? "")) {
        this.resumeTried = true; void this.resume(); return;      // the agent may have crashed: ask the API for a fresh one
      }
      if (s.state === "ended" || s.state === "error") this.ended = true;
      this.render();
    });
    document.body.appendChild(this.host);
    this.render();
  }

  toggle(force?: boolean) { this.open = force ?? !this.open; this.render(); }

  private async start() {
    if (this.busy) return;
    this.busy = true; this.msg = ""; this.ended = false; this.rated = false; this.resumeTried = false; this.render();
    try {
      const mic = await VoxaCall.checkMic();
      if (mic === "unsupported") throw new Error("Your browser does not support voice calls.");
      if (mic === "denied") throw new Error("Microphone access is blocked. Allow it in your browser's site settings, then try again.");
      const r = await fetch(`${cfg.api}/v1/widget/token`, { method: "POST", headers: { "X-Widget-Key": cfg.key } });
      if (r.status === 403) throw new Error("This site is not allowed to use the assistant.");
      if (!r.ok) throw new Error("Could not start the call. Please try again shortly.");
      const creds = await r.json();
      this.csatToken = creds.csat_token ?? "";
      await this.call.start(creds);
    } catch (e) {
      this.msg = (e as Error).message;
      this.ended = false;
    } finally { this.busy = false; this.render(); }
  }

  private async resume() {
    try {
      const r = await fetch(`${cfg.api}/v1/widget/resume`, { method: "POST", headers: { "content-type": "application/json" },
        body: JSON.stringify({ token: this.csatToken }) });
      if (!r.ok) throw new Error("not resumable");
      await this.call.start(await r.json());
    } catch { this.snap = { ...this.snap, state: "error", error: "Connection lost. Please start a new call." }; this.ended = true; this.render(); }
  }

  private render() {
    const s = this.snap;
    const inCall = !["idle", "ended", "error"].includes(s.state);
    const theme = cfg.theme === "dark" ? "dark" : cfg.theme === "light" ? "" : "auto";
    const caps = s.captions.map((c) =>
      `<p class="cap ${c.role}${c.final ? "" : " partial"}"><b>${c.role === "user" ? "You" : "Assistant"}</b>${esc(c.text)}</p>`).join("");
    let body = "";
    if (!inCall && !this.ended) {
      body = `<div class="center"><p style="margin:0 0 12px">${esc(cfg.prompt)}</p>
        <button class="btn primary" data-a="start" ${this.busy ? "disabled" : ""}>${this.busy ? "Starting…" : "Start call"}</button>
        ${this.msg ? `<p role="alert" style="color:#dc2626;font-size:13px">${esc(this.msg)}</p>` : ""}</div>
        <p class="note">Your voice is sent to this site's assistant. Calls may be transcribed.</p>`;
    } else if (inCall) {
      body = `<div class="stage s-${s.state}"><div class="orb" style="--lvl:${s.state === "speaking" ? s.agentLevel.toFixed(2) : 0}"></div></div>
        <div class="status" role="status">${LABEL[s.state]}${s.muted ? " · muted" : ""}</div>
        <div class="caps" aria-live="polite" aria-label="Live captions" id="caps">${caps || '<span style="color:var(--muted)">Captions appear here.</span>'}</div>
        <div class="row"><button class="btn" data-a="mute" aria-pressed="${s.muted}">${s.muted ? "Unmute" : "Mute"}</button>
        <button class="btn danger" data-a="end">End call</button></div>`;
    } else {
      body = `<div class="center">${s.state === "error" ? `<p role="alert" style="color:#dc2626">${esc(s.error ?? LABEL.error)}</p>` : "<p>Thanks for talking with us.</p>"}
        ${!this.rated && s.state === "ended" ? `<p style="margin:8px 0 4px;font-size:13px;color:var(--muted)">How was it?</p><div class="stars">${[1, 2, 3, 4, 5].map((n) => `<button data-a="rate" data-n="${n}" aria-label="${n} star${n > 1 ? "s" : ""}">★</button>`).join("")}</div>` : ""}
        ${this.rated ? "<p>Thank you for your feedback!</p>" : ""}
        <p><button class="btn primary" data-a="restart">${s.state === "error" ? "Try again" : "New call"}</button></p></div>`;
    }
    this.sh.innerHTML = `<style>${css}</style><div class="root ${theme}">
      <div class="panel ${this.open ? "open" : ""}" role="dialog" aria-label="${esc(cfg.title)}">
        <header><span>${esc(cfg.title)}</span><button class="x" data-a="close" aria-label="Close">×</button></header>${body}</div>
      <button class="fab" data-a="toggle" aria-label="${this.open ? "Close" : "Open"} voice assistant" aria-expanded="${this.open}">${mic}</button></div>`;
    const capsEl = this.sh.getElementById("caps");
    if (capsEl) capsEl.scrollTop = capsEl.scrollHeight;
    this.sh.querySelectorAll<HTMLElement>("[data-a]").forEach((el) => el.addEventListener("click", () => this.act(el)));
  }

  private async act(el: HTMLElement) {
    switch (el.dataset.a) {
      case "toggle": this.toggle(); break;
      case "close": this.toggle(false); break;
      case "start": case "restart": this.ended = false; this.snap = { ...this.snap, state: "idle" }; await this.start(); break;
      case "mute": await this.call.setMuted(!this.snap.muted); break;
      case "end": await this.call.end(); break;
      case "rate": {
        const score = Number(el.dataset.n);
        this.call.sendCsat(score);                       // also over the data channel if still connected
        if (this.csatToken) void fetch(`${cfg.api}/v1/widget/csat`, { method: "POST", headers: { "content-type": "application/json" },
          body: JSON.stringify({ token: this.csatToken, score }) }).catch(() => {});
        this.rated = true; this.render(); break;
      }
    }
  }
}

function esc(s: string): string {
  return s.replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" })[c]!);
}

if (!cfg.key) console.warn("[voxa] widget: missing data-key");
else {
  const w = new Widget();
  (window as any).Voxa = { open: () => w.toggle(true), close: () => w.toggle(false) };
}
