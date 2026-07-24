"""Ask-ARISE: an offline retrieval (RAG) assistant over the project knowledge base.

Design (matches the offline-first requirement):

* **Index, not training.** Documents are split into overlapping chunks and
  indexed with a pure-numpy TF-IDF matrix -- no network, no GPU, no model
  download. Retrieval is cosine similarity; answers are extractive (the most
  relevant passages, stitched and cited).
* **Sources**: the project docs (README, docs/RESEARCH.md), the latest run's
  QA summary + candidate list (so you can ask about *your own data*), plus any
  files or URLs the user drops into ``kb/`` or lists in ``kb/urls.txt``.
* **Optional generation**: if ANTHROPIC_API_KEY is set and the ``anthropic``
  package is installed, retrieved passages are handed to Claude for a fluent
  answer. Entirely optional -- everything works without it.
"""
from __future__ import annotations

import ipaddress
import json
import math
import re
import socket
import time
from dataclasses import dataclass, field
from html.parser import HTMLParser
from pathlib import Path
from urllib.parse import urljoin, urlparse

import numpy as np

from .logs import get_logger

log = get_logger("rag")

_WORD = re.compile(r"[a-z0-9][a-z0-9_.\-]{1,}")
_STOP = set("""a an and are as at be but by for from has have if in into is it its of on or
that the this to was were will with what which how why when where who your you we our
i me my they them their he she his her do does did not no yes can could should would
""".split())


def _stem(t: str) -> str:
    """Light suffix stripping so 'movers'/'moving', 'transients'/'transient',
    'discovered'/'discovers'/'discovery' land on one token. Suffixes are tried
    longest-first and stripped repeatedly to a fixpoint (plus a trailing e/y
    rule), so chained forms co-tokenize. Crude but dependency-free."""
    while True:
        for suf in ("ing", "ers", "ies", "ed", "es", "er", "s"):
            if t.endswith(suf) and len(t) - len(suf) >= 3:
                t = t[: len(t) - len(suf)]
                break
        else:
            if len(t) >= 5 and t[-1] in "ey":   # discovery->discover, source(s)->sourc
                t = t[:-1]
                continue
            return t


def _tokenize(text: str) -> list[str]:
    return [_stem(t) for t in _WORD.findall(text.lower()) if t not in _STOP]


class _HTMLText(HTMLParser):
    """Minimal HTML -> text (keeps us dependency-free for URL ingestion)."""

    _SKIP = {"script", "style", "nav", "footer", "header"}

    def __init__(self):
        super().__init__()
        self._skip = 0
        self.parts: list[str] = []

    def handle_starttag(self, tag, attrs):
        if tag in self._SKIP:
            self._skip += 1

    def handle_endtag(self, tag):
        if tag in self._SKIP and self._skip:
            self._skip -= 1

    def handle_data(self, data):
        if not self._skip and data.strip():
            self.parts.append(data.strip())


def _html_to_text(html: str) -> str:
    p = _HTMLText()
    try:
        p.feed(html)
    except Exception:
        return html
    return "\n".join(p.parts)


_KB_URL_MAX_BYTES = 2_000_000     # hard cap on bytes fetched per URL


def _url_blocked_reason(url: str) -> str | None:
    """SSRF guard: return why *url* must not be fetched, or None if it looks OK.

    Only http(s) to hosts resolving exclusively to public addresses is allowed;
    loopback, private, link-local (cloud metadata), reserved, multicast and
    unspecified targets are refused. Dependency-free (socket + ipaddress).
    """
    parsed = urlparse(url)
    if parsed.scheme not in ("http", "https"):
        return f"scheme {parsed.scheme!r} not allowed"
    host = parsed.hostname
    if not host:
        return "no host in URL"
    try:
        port = parsed.port or (443 if parsed.scheme == "https" else 80)
    except ValueError:
        return "invalid port"
    try:
        infos = socket.getaddrinfo(host, port, proto=socket.IPPROTO_TCP)
    except OSError as exc:
        return f"cannot resolve host ({exc})"
    for info in infos:
        try:
            addr = ipaddress.ip_address(info[4][0])
        except ValueError:
            return f"unparseable resolved address {info[4][0]!r}"
        if isinstance(addr, ipaddress.IPv6Address) and addr.ipv4_mapped:
            addr = addr.ipv4_mapped   # e.g. ::ffff:169.254.169.254
        if (addr.is_loopback or addr.is_private or addr.is_link_local
                or addr.is_reserved or addr.is_multicast or addr.is_unspecified):
            return f"host resolves to non-public address {addr}"
    return None


# --------------------------------------------------------------------------- #
@dataclass
class Chunk:
    source: str
    text: str


@dataclass
class KnowledgeBase:
    chunks: list[Chunk] = field(default_factory=list)
    _vocab: dict[str, int] = field(default_factory=dict)
    _matrix: np.ndarray | None = None      # (n_chunks, vocab) L2-normalised tf-idf
    _idf: np.ndarray | None = None

    # ---- ingestion ------------------------------------------------------ #
    def add_text(self, source: str, text: str, chunk_words: int = 220,
                 overlap: int = 40) -> int:
        """Chunk + register a document. Returns number of chunks added."""
        text = re.sub(r"\n{3,}", "\n\n", text or "").strip()
        if not text:
            return 0
        words = text.split()
        n = 0
        step = max(chunk_words - overlap, 50)
        for i in range(0, len(words), step):
            piece = " ".join(words[i:i + chunk_words]).strip()
            if len(piece) > 80:
                self.chunks.append(Chunk(source=source, text=piece))
                n += 1
        self._matrix = None      # invalidate index
        return n

    def add_file(self, path: str | Path) -> int:
        path = Path(path)
        if not path.exists():
            return 0
        suffix = path.suffix.lower()
        try:
            if suffix in (".md", ".txt", ".py", ".yaml", ".yml", ".rst", ".log"):
                return self.add_text(path.name, path.read_text(encoding="utf-8", errors="replace"))
            if suffix == ".json":
                data = json.loads(path.read_text(encoding="utf-8", errors="replace"))
                return self.add_text(path.name, json.dumps(data, indent=1)[:60000])
            if suffix == ".csv":
                txt = path.read_text(encoding="utf-8", errors="replace")
                lines = txt.splitlines()
                # keep the header with every chunk so rows stay interpretable
                head = lines[0] if lines else ""
                body = "\n".join(lines[1:400])
                return self.add_text(path.name, f"CSV {path.name}\ncolumns: {head}\n{body}")
            if suffix in (".html", ".htm"):
                return self.add_text(path.name, _html_to_text(
                    path.read_text(encoding="utf-8", errors="replace")))
        except Exception as exc:
            log.warning("KB skipped %s (%s)", path.name, exc)
        return 0

    def add_folder(self, folder: str | Path, max_files: int = 400,
                   max_bytes: int = 8_000_000) -> int:
        """Ingest every note in a folder tree (e.g. an Obsidian vault).

        Vaults are plain folders of markdown, so no plugin or export is needed.
        Caps on file count / total bytes keep huge vaults from bloating the index.
        """
        folder = Path(folder).expanduser()
        if not folder.is_dir():
            log.warning("KB folder not found: %s", folder)
            return 0
        total_chunks = total_bytes = n_files = 0
        for f in sorted(folder.rglob("*")):
            if n_files >= max_files or total_bytes >= max_bytes:
                log.warning("KB folder %s truncated at %d files / %d bytes",
                            folder.name, n_files, total_bytes)
                break
            if not f.is_file() or f.suffix.lower() not in (".md", ".txt"):
                continue
            if any(part.startswith(".") for part in f.relative_to(folder).parts):   # .obsidian etc.
                continue
            try:
                text = f.read_text(encoding="utf-8", errors="replace")
            except OSError:
                continue
            total_bytes += len(text)
            n_files += 1
            total_chunks += self.add_text(f"{folder.name}/{f.relative_to(folder)}", text)
        log.info("KB ingested folder %s: %d notes -> %d chunks", folder, n_files, total_chunks)
        return total_chunks

    def add_url(self, url: str, timeout: int = 20) -> int:
        """Fetch a URL (API docs page, reference, etc.) into the KB.

        Only public http(s) hosts are fetched (SSRF guard, re-checked on every
        redirect hop), and the body is streamed with a hard byte cap and a
        wall-clock deadline so an oversized or trickling response cannot
        exhaust memory or stall knowledge-base construction.
        """
        try:
            import requests
            r = None
            for _hop in range(4):          # original URL + up to 3 redirects
                reason = _url_blocked_reason(url)
                if reason:
                    log.warning("KB refused to fetch %s (%s)", url, reason)
                    return 0
                r = requests.get(url, timeout=(5, timeout), stream=True,
                                 allow_redirects=False,
                                 headers={"User-Agent": "ARISE-KB/0.1"})
                if not r.is_redirect:
                    break
                target = r.headers.get("location", "")
                r.close()
                r = None
                url = urljoin(url, target)
            if r is None:
                log.warning("KB could not fetch %s (too many redirects)", url)
                return 0
            with r:
                r.raise_for_status()
                clen = r.headers.get("content-length", "")
                if clen.isdigit() and int(clen) > _KB_URL_MAX_BYTES:
                    log.warning("KB refused %s (declared %s bytes > %d-byte cap)",
                                url, clen, _KB_URL_MAX_BYTES)
                    return 0
                buf = bytearray()
                deadline = time.monotonic() + 60.0
                for chunk in r.iter_content(chunk_size=65536):
                    buf += chunk
                    if len(buf) >= _KB_URL_MAX_BYTES or time.monotonic() > deadline:
                        log.warning("KB truncated %s at %d bytes", url, len(buf))
                        break
                ctype = r.headers.get("content-type", "")
                encoding = r.encoding
            text = bytes(buf).decode(encoding or "utf-8", errors="replace")
            if "html" in ctype:
                text = _html_to_text(text)
            n = self.add_text(url, text[:120000])
            log.info("KB ingested %s (%d chunks)", url, n)
            return n
        except Exception as exc:
            log.warning("KB could not fetch %s (%s)", url, exc)
            return 0

    # ---- index ----------------------------------------------------------- #
    def _build(self) -> None:
        docs = [_tokenize(c.text) for c in self.chunks]
        vocab: dict[str, int] = {}
        for toks in docs:
            for t in toks:
                if t not in vocab:
                    vocab[t] = len(vocab)
        if not vocab:
            self._vocab, self._matrix, self._idf = {}, np.zeros((0, 0)), np.zeros(0)
            return
        n_docs = len(docs)
        df = np.zeros(len(vocab))
        rows = []
        for toks in docs:
            counts: dict[int, float] = {}
            for t in toks:
                counts[vocab[t]] = counts.get(vocab[t], 0.0) + 1.0
            for j in counts:
                df[j] += 1
            rows.append(counts)
        idf = np.log((1.0 + n_docs) / (1.0 + df)) + 1.0
        mat = np.zeros((n_docs, len(vocab)), dtype=np.float32)
        for i, counts in enumerate(rows):
            for j, c in counts.items():
                mat[i, j] = (1.0 + math.log(c)) * idf[j]
            norm = np.linalg.norm(mat[i])
            if norm > 0:
                mat[i] /= norm
        self._vocab, self._matrix, self._idf = vocab, mat, idf

    # ---- query ----------------------------------------------------------- #
    def search(self, question: str, k: int = 4) -> list[tuple[Chunk, float]]:
        if self._matrix is None:
            self._build()
        if self._matrix is None or not self._matrix.size:
            return []
        q = np.zeros(len(self._vocab), dtype=np.float32)
        toks = _tokenize(question)
        if not toks:
            return []
        for t in toks:
            j = self._vocab.get(t)
            if j is not None:
                q[j] += self._idf[j]
        norm = np.linalg.norm(q)
        if norm == 0:
            return []
        q /= norm
        sims = self._matrix @ q
        order = np.argsort(sims)[::-1][:k]
        return [(self.chunks[i], float(sims[i])) for i in order if sims[i] > 0.03]

    def answer(self, question: str, k: int = 6) -> dict:
        """Answer a question. Extractive by default; generative if a key exists."""
        hits = self.search(question, k=k)
        if not hits:
            return {"answer": "I could not find anything relevant in the knowledge "
                              "base. Add documents or URLs to kb/ and try again.",
                    "sources": [], "mode": "none"}
        passages = [{"source": c.source, "score": round(s, 3),
                     "text": c.text[:900]} for c, s in hits]

        generated = _generate_with_claude(question, passages)
        if generated:
            return {"answer": generated, "sources": passages, "mode": "generative"}
        generated = _generate_with_nvidia(question, passages)
        if generated:
            return {"answer": generated, "sources": passages, "mode": "generative"}

        # extractive fallback: stitch the best passages with their citations
        lines = []
        for i, p in enumerate(passages[:3], 1):
            lines.append(f"[{i}] ({p['source']}) {p['text']}")
        return {"answer": "\n\n".join(lines), "sources": passages, "mode": "extractive"}


def _generate_with_claude(question: str, passages: list[dict]) -> str | None:
    """Optional: fluent answer via the Claude API when a key is configured."""
    import os
    if not os.environ.get("ANTHROPIC_API_KEY"):
        return None
    try:
        import anthropic
        client = anthropic.Anthropic()
        ctx = "\n\n---\n\n".join(f"SOURCE {i+1} ({p['source']}):\n{p['text']}"
                                 for i, p in enumerate(passages))
        msg = client.messages.create(
            model="claude-haiku-4-5-20251001",
            max_tokens=600,
            messages=[{"role": "user", "content":
                       f"Answer the question using ONLY these sources; cite them as "
                       f"[1], [2]... If they don't contain the answer, say so.\n\n"
                       f"{ctx}\n\nQUESTION: {question}"}],
        )
        return msg.content[0].text
    except Exception as exc:
        log.warning("Claude generation unavailable (%s); using extractive answer", exc)
        return None


_RAG_PROMPT = ("Answer the question using ONLY these sources; cite them as [1], [2]... "
               "Be concise and factual. If the sources don't contain the answer, say so.\n\n"
               "{ctx}\n\nQUESTION: {question}")


def _generate_with_nvidia(question: str, passages: list[dict]) -> str | None:
    """Fluent answer via NVIDIA-hosted LLMs; rotates keys on rate limits."""
    import os
    keys = [k for k in os.environ.get("NVIDIA_API_KEYS", "").split(",") if k.strip()]
    if not keys:
        return None
    try:
        import requests
        ctx = "\n\n---\n\n".join(f"SOURCE {i+1} ({p['source']}):\n{p['text']}"
                                 for i, p in enumerate(passages))
        body = {"model": "meta/llama-3.1-70b-instruct",
                "messages": [{"role": "user",
                              "content": _RAG_PROMPT.format(ctx=ctx, question=question)}],
                "max_tokens": 600, "temperature": 0.2}
        for key in keys:
            r = requests.post("https://integrate.api.nvidia.com/v1/chat/completions",
                              headers={"Authorization": f"Bearer {key.strip()}"},
                              json=body, timeout=45)
            if r.status_code == 429:      # rate-limited -> try the next key
                continue
            r.raise_for_status()
            return r.json()["choices"][0]["message"]["content"].strip()
    except Exception as exc:
        log.warning("NVIDIA generation unavailable (%s); using extractive answer", exc)
    return None


# --------------------------------------------------------------------------- #
def build_default_kb(project_root: str | Path, data_dirs: list[str | Path] = ()) -> KnowledgeBase:
    """Standard ARISE knowledge base: docs + configs + latest run outputs + kb/."""
    root = Path(project_root)
    kb = KnowledgeBase()
    for rel in ("README.md", "docs/RESEARCH.md"):
        kb.add_file(root / rel)
    for cfg in sorted((root / "config").glob("*.yaml")) if (root / "config").exists() else []:
        if "key" in cfg.name.lower() or "secret" in cfg.name.lower():
            continue   # never index credential files (keys.yaml, secrets.yaml, ...)
        kb.add_file(cfg)

    # latest pipeline outputs (lets you ask questions about your own night)
    for d in data_dirs:
        d = Path(d)
        for name in ("reports/run_summary.txt", "reports/night_brief.md",
                     "reports/qa_summary.json", "catalogs/candidates.csv"):
            kb.add_file(d / name)

    # user-supplied knowledge: files in kb/, URLs in kb/urls.txt, and whole
    # folders (e.g. an Obsidian vault -- it is just a folder of .md files)
    # listed one-per-line in kb/sources.txt
    kb_dir = root / "kb"
    if kb_dir.exists():
        for f in sorted(kb_dir.iterdir()):
            if f.name == "urls.txt":
                for line in f.read_text(encoding="utf-8-sig", errors="replace").splitlines():
                    url = line.strip()
                    if url and not url.startswith("#"):
                        kb.add_url(url)
            elif f.name == "sources.txt":
                for line in f.read_text(encoding="utf-8-sig", errors="replace").splitlines():
                    src = line.strip().strip('"')
                    if src and not src.startswith("#"):
                        kb.add_folder(src)
            elif f.is_file():
                kb.add_file(f)

    log.info("Knowledge base ready: %d chunks from %d sources",
             len(kb.chunks), len({c.source for c in kb.chunks}))
    return kb
