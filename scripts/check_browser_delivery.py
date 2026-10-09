"""Run real Chromium against a localhost atlas and its cold offline file package."""
from pathlib import Path
import argparse
import hashlib
import json
import os
import subprocess
import sys
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from threading import Thread

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--node", default="node")
    args = parser.parse_args()
    module = Path(os.environ.get("PLAYWRIGHT_PATH", ROOT / "tools/browser/node_modules/playwright")).resolve()
    package = module / "package.json"
    if not package.is_file():
        print("UNVERIFIED: pinned Playwright missing. Run npm ci --prefix tools/browser, then node tools/browser/node_modules/playwright/cli.js install chromium", file=sys.stderr)
        return 2
    if json.loads(package.read_text(encoding="utf-8"))["version"] != "1.62.1":
        raise ValueError("Required Playwright version is 1.62.1")
    browser_check = subprocess.run([args.node, "-e", "const p=require(process.argv[1]); const f=require('fs'); const e=process.env.CHROME_PATH||p.chromium.executablePath(); if(!f.existsSync(e))process.exit(2); process.stdout.write(e)", str(module)], capture_output=True, text=True)
    if browser_check.returncode:
        print("UNVERIFIED: Chromium runtime missing. Run node tools/browser/node_modules/playwright/cli.js install chromium", file=sys.stderr)
        return 2
    executable = Path(browser_check.stdout)
    output = ROOT / "artifacts/browser-delivery"
    output.mkdir(parents=True, exist_ok=True)
    (output / "environment.json").write_text(json.dumps({"playwright": "1.62.1", "chromium_executable": str(executable),
        "chromium_executable_sha256": hashlib.sha256(executable.read_bytes()).hexdigest(), "scope": "Local Chromium emulation, not physical-device certification"}, indent=2) + "\n", encoding="utf-8")
    server = ThreadingHTTPServer(("127.0.0.1", 0), partial(SimpleHTTPRequestHandler, directory=str(ROOT)))
    thread = Thread(target=server.serve_forever, daemon=True)
    thread.start()
    env = dict(os.environ, PLAYWRIGHT_PATH=str(module), TEST_BASE_URL=f"http://127.0.0.1:{server.server_port}/docs/index.html")
    codes = []
    try:
        for script, folder in (("tests/browser/final_atlas_checks.cjs", "main"), ("tools/browser/offline.cjs", "offline")):
            codes.append(subprocess.run([args.node, script, "docs/index.html", str(output / folder)], cwd=ROOT, env=env).returncode)
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)
    return 1 if any(codes) else 0


if __name__ == "__main__":
    raise SystemExit(main())
