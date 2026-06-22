#!/usr/bin/env python3
from __future__ import annotations

import http.server
import socketserver
import sys
from urllib import error, parse, request


UPSTREAM = "https://www.openstreetmap.org"
TILE_UPSTREAM = "https://tile.openstreetmap.org"
HOP_BY_HOP = {
    "connection",
    "keep-alive",
    "proxy-authenticate",
    "proxy-authorization",
    "te",
    "trailers",
    "transfer-encoding",
    "upgrade",
}
DROP_HEADERS = {
    "content-encoding",
    "content-length",
    "content-security-policy",
    "strict-transport-security",
    "set-cookie",
}


class OSMProxy(http.server.BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def do_GET(self) -> None:
        self._proxy()

    def do_HEAD(self) -> None:
        self._proxy(head_only=True)

    def do_POST(self) -> None:
        self._proxy()

    def _proxy(self, head_only: bool = False) -> None:
        if self.path.startswith("/tile/"):
            upstream = TILE_UPSTREAM
            target_path = self.path[len("/tile") :]
        else:
            upstream = UPSTREAM
            target_path = self.path
        target = upstream + target_path
        data = None
        if self.command == "POST":
            length = int(self.headers.get("Content-Length", "0") or "0")
            data = self.rfile.read(length) if length else b""

        headers = {
            "User-Agent": self.headers.get("User-Agent", "Mozilla/5.0"),
            "Accept": self.headers.get("Accept", "*/*"),
            "Accept-Language": self.headers.get("Accept-Language", "en-US,en;q=0.9"),
            "Accept-Encoding": "identity",
            "Host": parse.urlparse(upstream).netloc,
            "Origin": upstream,
            "Referer": upstream + "/",
        }
        req = request.Request(target, data=data, headers=headers, method=self.command)
        try:
            with request.urlopen(req, timeout=30) as resp:
                body = b"" if head_only else resp.read()
                body = self._rewrite_body(body, resp.headers.get("Content-Type", ""))
                self.send_response(resp.status)
                for key, value in resp.headers.items():
                    lower = key.lower()
                    if lower in HOP_BY_HOP or lower in DROP_HEADERS:
                        continue
                    if lower == "location":
                        value = self._rewrite_url(value)
                    self.send_header(key, value)
                self.send_header("Content-Length", str(len(body)))
                self.send_header("Access-Control-Allow-Origin", "*")
                self.end_headers()
                if not head_only:
                    self.wfile.write(body)
        except error.HTTPError as exc:
            body = b"" if head_only else exc.read()
            self.send_response(exc.code)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            if not head_only:
                self.wfile.write(body)
        except Exception as exc:
            body = f"OSM proxy error: {exc!r}\n".encode()
            self.send_response(502)
            self.send_header("Content-Type", "text/plain; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            if not head_only:
                self.wfile.write(body)

    def _rewrite_url(self, value: str) -> str:
        parsed = parse.urlparse(value)
        if parsed.netloc == "www.openstreetmap.org":
            return parse.urlunparse(("", "", parsed.path, parsed.params, parsed.query, parsed.fragment)) or "/"
        return value

    def _rewrite_body(self, body: bytes, content_type: str) -> bytes:
        if "text/html" not in content_type and "text/css" not in content_type and "javascript" not in content_type:
            return body
        text = body.decode("utf-8", errors="replace")
        text = text.replace("https://www.openstreetmap.org", "")
        text = text.replace("http://www.openstreetmap.org", "")
        return text.encode("utf-8")

    def log_message(self, fmt: str, *args: object) -> None:
        sys.stderr.write("%s - - [%s] %s\n" % (self.address_string(), self.log_date_time_string(), fmt % args))


def main() -> None:
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 3000
    with socketserver.ThreadingTCPServer(("0.0.0.0", port), OSMProxy) as httpd:
        httpd.allow_reuse_address = True
        print(f"Serving OSM proxy on http://0.0.0.0:{port} -> {UPSTREAM}", flush=True)
        httpd.serve_forever()


if __name__ == "__main__":
    main()
