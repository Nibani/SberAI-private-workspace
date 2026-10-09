"""Read atlas data from standalone HTML or its local offline bundle without executing JavaScript."""
import base64
import gzip
import hashlib
from html.parser import HTMLParser
import json
from pathlib import Path
import re


class _ScriptSources(HTMLParser):
    def __init__(self):
        super().__init__()
        self.sources = []

    def handle_starttag(self, tag, attrs):
        if tag == "script":
            self.sources.extend(value for name, value in attrs if name == "src" and value is not None)


def read_atlas_payload(path: Path) -> dict:
    """Accept JSON literals and hash-checked assets confined to the sibling assets directory."""
    text = path.read_text(encoding="utf-8")
    inline = re.search(r"\bconst\s+packed\s*=\s*(?=\{)", text)
    if inline:
        packed, end = json.JSONDecoder().raw_decode(text[inline.end():])
        if not text[inline.end() + end:].lstrip().startswith(";"):
            raise ValueError("Invalid inline atlas data package")
    else:
        parser = _ScriptSources()
        parser.feed(text)
        sources = [source for source in parser.sources
                   if re.fullmatch(r"assets/payload-[0-9a-f]{64}\.js", source)]
        if len(sources) != 1:
            raise ValueError("Atlas data package not found: expected one local hash-named asset")
        parent = path.parent.resolve()
        directory = (parent / "assets").resolve()
        asset = (parent / sources[0]).resolve()
        if directory != parent / "assets" or asset.parent != directory:
            raise ValueError("Atlas asset must stay within the sibling assets directory")
        raw = asset.read_bytes()
        expected = Path(sources[0]).stem.removeprefix("payload-")
        if hashlib.sha256(raw).hexdigest() != expected:
            raise ValueError("Atlas asset SHA256 differs from its filename")
        assignment = re.fullmatch(r'"use strict";\s*window\.__SBERAI_ATLAS__=\{payload:(.*)\};\s*',
                                  raw.decode("utf-8"), re.S)
        if assignment is None:
            raise ValueError("Invalid classic atlas data script")
        packed = json.loads(assignment.group(1))
    if not isinstance(packed, dict):
        raise ValueError("Unknown atlas data encoding")
    if packed.get("encoding") == "gzip-base64":
        packed = json.loads(gzip.decompress(base64.b64decode(packed["data"], validate=True)))
    if not isinstance(packed, dict) or "entities" not in packed:
        raise ValueError("Unknown atlas data encoding")
    return packed
