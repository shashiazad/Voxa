"""Document parsers -> plain text / FAQ pairs. Connectors implement `Connector`."""
from __future__ import annotations

import csv
import io
import re
from abc import ABC, abstractmethod
from dataclasses import dataclass
from urllib.parse import urljoin, urlparse

import httpx


@dataclass
class Doc:
    title: str
    source: str
    text: str = ""
    faq: list[tuple[str, str]] | None = None


def parse_pdf(data: bytes, name: str) -> Doc:
    from pypdf import PdfReader

    r = PdfReader(io.BytesIO(data))
    return Doc(name, name, "\n\n".join((p.extract_text() or "") for p in r.pages))


def parse_docx(data: bytes, name: str) -> Doc:
    from docx import Document

    d = Document(io.BytesIO(data))
    parts = [p.text for p in d.paragraphs if p.text.strip()]
    for t in d.tables:
        for row in t.rows:
            parts.append(" | ".join(c.text.strip() for c in row.cells))
    return Doc(name, name, "\n\n".join(parts))


def parse_csv(data: bytes, name: str) -> Doc:
    """FAQ CSV: columns question/answer (case-insensitive, also q/a). Otherwise rows become text."""
    rows = list(csv.DictReader(io.StringIO(data.decode("utf-8-sig"))))
    if rows:
        keys = {k.lower().strip(): k for k in rows[0]}
        qk = keys.get("question") or keys.get("q")
        ak = keys.get("answer") or keys.get("a")
        if qk and ak:
            return Doc(name, name, faq=[(r[qk], r[ak]) for r in rows])
    return Doc(name, name, "\n".join("; ".join(f"{k}: {v}" for k, v in r.items()) for r in rows))


def parse_text(data: bytes, name: str) -> Doc:
    return Doc(name, name, data.decode("utf-8", errors="replace"))


def html_to_text(html: str) -> tuple[str, str]:
    from bs4 import BeautifulSoup

    soup = BeautifulSoup(html, "html.parser")
    for t in soup(["script", "style", "nav", "footer", "header", "aside", "form", "noscript"]):
        t.decompose()
    title = (soup.title.string or "").strip() if soup.title and soup.title.string else ""
    text = re.sub(r"\n{3,}", "\n\n", soup.get_text("\n"))
    return title, re.sub(r"[ \t]+", " ", text).strip()


async def crawl_url(url: str, max_pages: int = 20, same_host: bool = True,
                    client: httpx.AsyncClient | None = None) -> list[Doc]:
    """Breadth-first crawl limited to the same host. Only http(s); no private-network targets
    (SSRF guard) unless the caller passes their own client for tests."""
    own = client is None
    client = client or httpx.AsyncClient(timeout=10, follow_redirects=True, headers={"User-Agent": "VoxaBot/0.1"})
    host = urlparse(url).netloc
    seen, queue, docs = set(), [url], []
    try:
        while queue and len(docs) < max_pages:
            u = queue.pop(0)
            if u in seen or urlparse(u).scheme not in ("http", "https"):
                continue
            seen.add(u)
            if own and _is_private(urlparse(u).hostname or ""):
                continue
            try:
                r = await client.get(u)
            except httpx.HTTPError:
                continue
            if "text/html" not in r.headers.get("content-type", ""):
                continue
            title, text = html_to_text(r.text)
            if text:
                docs.append(Doc(title or u, u, text))
            from bs4 import BeautifulSoup

            for a in BeautifulSoup(r.text, "html.parser").find_all("a", href=True):
                nxt = urljoin(u, a["href"]).split("#")[0]
                if not same_host or urlparse(nxt).netloc == host:
                    queue.append(nxt)
    finally:
        if own:
            await client.aclose()
    return docs


def _is_private(host: str) -> bool:
    import ipaddress
    import socket

    try:
        infos = socket.getaddrinfo(host, None)
    except OSError:
        return True
    for i in infos:
        ip = ipaddress.ip_address(i[4][0])
        if ip.is_private or ip.is_loopback or ip.is_link_local or ip.is_reserved:
            return True
    return False


class Connector(ABC):
    """Pluggable source (Zendesk, Notion, Confluence...). Yields Docs; the pipeline handles the rest."""
    name: str

    @abstractmethod
    async def fetch(self) -> list[Doc]: ...


class ZendeskHelpCenter(Connector):
    """Zendesk Help Center articles via the public REST API. NOT exercised against a live account."""
    name = "zendesk"

    def __init__(self, subdomain: str, locale: str = "en-us", client: httpx.AsyncClient | None = None) -> None:
        self.base = f"https://{subdomain}.zendesk.com/api/v2/help_center/{locale}/articles.json"
        self._client = client

    async def fetch(self) -> list[Doc]:
        client = self._client or httpx.AsyncClient(timeout=15)
        docs, url = [], self.base
        try:
            while url and len(docs) < 1000:
                r = await client.get(url)
                r.raise_for_status()
                j = r.json()
                for a in j.get("articles", []):
                    if a.get("draft"):
                        continue
                    _, text = html_to_text(a.get("body") or "")
                    docs.append(Doc(a.get("title", "article"), a.get("html_url", url), text))
                url = j.get("next_page")
        finally:
            if self._client is None:
                await client.aclose()
        return docs
