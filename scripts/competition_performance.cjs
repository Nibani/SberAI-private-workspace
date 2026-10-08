"use strict";

// Freeze the experiment before timing and compare the actual atlas functions.
const fs = require("node:fs");
const path = require("node:path");
const crypto = require("node:crypto");
const os = require("node:os");
const assert = require("node:assert/strict");
const { performance } = require("node:perf_hooks");
const { execFileSync } = require("node:child_process");

const ROOT = path.resolve(__dirname, "..");
const OUT = path.join(ROOT, "reports/competition-enhancement/delivery/performance-new");
const sha = bytes => crypto.createHash("sha256").update(bytes).digest("hex");
const readJSON = file => JSON.parse(fs.readFileSync(file, "utf8"));
const saveJSON = (file, value) => fs.writeFileSync(file, JSON.stringify(value, null, 2) + "\n");
const relative = file => path.relative(ROOT, file).replaceAll("\\", "/");
const median = values => {
  const sorted = values.slice().sort((a, b) => a - b), middle = Math.floor(sorted.length / 2);
  return sorted.length % 2 ? sorted[middle] : (sorted[middle - 1] + sorted[middle]) / 2;
};
const functionLine = (source, name) => {
  const found = source.match(new RegExp("^function " + name + "\\([^\\n]+", "m"));
  if (!found) throw new Error("Function missing: " + name);
  return found[0].replace(/\r$/, "");
};
const SELECT = `function nearestRows(rows,count,compare){
 if(!Number.isSafeInteger(count)||count<0||rows.some(row=>Number.isNaN(row.d)))return rows.sort(compare).slice(0,count);
 count=Math.min(count,rows.length);if(count===0)return [];
 const best=[];
 for(const row of rows){
  if(best.length===count&&compare(row,best[best.length-1])>=0)continue;
  let low=0,high=best.length;
  while(low<high){const middle=(low+high)>>>1;if(compare(row,best[middle])<0)high=middle;else low=middle+1}
  best.splice(low,0,row);if(best.length>count)best.pop();
 }
 return best;
}`;

function prepare() {
  fs.mkdirSync(OUT, { recursive: true });
  const planFile = path.join(OUT, "plan.json");
  if (fs.existsSync(planFile)) throw new Error("The preregistered plan already exists; do not overwrite it.");
  const templateFile = path.join(ROOT, "web/research_template.html");
  const templateBytes = fs.readFileSync(templateFile), template = templateBytes.toString("utf8");
  const htmlFile = path.join(ROOT, "docs/index.html"), html = fs.readFileSync(htmlFile);
  const assetMatch = html.toString("utf8").match(/<script src="(assets\/payload-[a-f0-9]+\.js)"><\/script>/);
  if (!assetMatch) throw new Error("Expected one immutable offline payload asset.");
  const assetFile = path.join(ROOT, "docs", assetMatch[1]), asset = fs.readFileSync(assetFile);
  const six = functionLine(template, "sixNeighbors"), nearest = functionLine(template, "nearest");
  const baseline = six + "\n" + nearest + "\n";
  const replacementSix = six.replace("return D.entities.map((b,i)=>", "return nearestRows(D.entities.map((b,i)=>")
    .replace(".sort((a,b)=>a.d-b.d||a.i-b.i).slice(0,contest.v12.neighbors)", ",contest.v12.neighbors,(a,b)=>a.d-b.d||a.i-b.i)");
  const replacementNearest = nearest.replace("return D.entities.map((b,i)=>", "return nearestRows(D.entities.map((b,i)=>")
    .replace(".sort((x,y)=>x.d-y.d||D.entities[x.i].id.localeCompare(D.entities[y.i].id)).slice(0,15)", ",15,(x,y)=>x.d-y.d||D.entities[x.i].id.localeCompare(D.entities[y.i].id))");
  assert.notEqual(replacementSix, six); assert.notEqual(replacementNearest, nearest);
  const candidate = SELECT + "\n" + replacementSix + "\n" + replacementNearest + "\n";
  const files = {
    "baseline-template.html": templateBytes,
    "baseline-functions.js": Buffer.from(baseline),
    "candidate-functions.js": Buffer.from(candidate),
  };
  for (const [name, bytes] of Object.entries(files)) fs.writeFileSync(path.join(OUT, name), bytes);
  const inputRecords = {
    [relative(assetFile)]: { sha256: sha(asset), bytes: asset.length },
    ...Object.fromEntries(Object.entries(files).map(([name, bytes]) => [relative(path.join(OUT, name)), { sha256: sha(bytes), bytes: bytes.length }])),
    [relative(__filename)]: { sha256: sha(fs.readFileSync(__filename)), bytes: fs.statSync(__filename).size },
  };
  const plan = {
    registered_at_utc: new Date().toISOString(),
    base_commit: execFileSync("git", ["rev-parse", "HEAD"], { cwd: ROOT, encoding: "utf8" }).trim(),
    node: process.version, node_executable: process.execPath,
    node_executable_sha256: sha(fs.readFileSync(process.execPath)),
    original_template_sha256: sha(templateBytes), original_html_sha256: sha(html),
    payload: relative(assetFile), frozen_inputs: inputRecords,
    timings_seen_before_registration: false,
    hypothesis: "Maintaining an ordered top-15 reduces exact nearest-neighbor selection time without changing any distance, tie order or self-exclusion behavior.",
    known_prior_result: "The deployed atob + byte loop already improved over Uint8Array.from(atob, callback) on an older payload. This is a fresh replication on the exact current payload, not a new decoder optimization.",
    repeats: 8, pair_order: ["AB", "BA", "AB", "BA", "AB", "BA", "AB", "BA"], warmup_batches: 1,
    scenario: { territory_index: "(query * 97) % entity_count", queries_per_batch: 240,
      functions: ["sixNeighbors", "nearest"], excluded: "sixNeighbors: query % 6; nearest: query % 5" },
    correctness: "DeepEqual all 2016 territories, sixNeighbors exclusions -1..5 and nearest exclusions -1..4; no rounding, unchanged sqrt and comparator. Synthetic ties, empty/singleton rows, zero/oversized/missing/negative count, NaN distances and missing-feature exceptions.",
    decode: "One warmup and eight AB/BA pairs. Measure base64, gzip streaming + UTF8, JSON.parse and their sum separately. Compare gzip byte SHA, decoded text SHA and DeepEqual on every call. No filesystem read inside timings.",
    decision: "Integrate top-15 only if exhaustive equivalence passes and all eight paired scenario times improve. Report raw ranges and medians; do not infer browser or CI speed.",
    resources: { threads: 1, initial_priority: "BelowNormal", required_external_monitor: "CPU, every disk, GPU and available RAM; reduce own load at 95%" },
    command: "node --single-threaded --v8-pool-size=1 --max-old-space-size=512 scripts/competition_performance.cjs measure reports/competition-enhancement/delivery/performance-new/plan.json",
    unknown: ["Browser map paint", "DOM rendering", "Cold/warm HTTPS transfer comparison", "Paired CI wall time", "Full offline build time improvement", "Native fromBase64 timing: absent in this Node runtime"],
  };
  saveJSON(planFile, plan);
  console.log(JSON.stringify({ status: "PREREGISTERED", plan: relative(planFile), sha256: sha(fs.readFileSync(planFile)), payload_sha256: sha(asset) }));
}

function makeFunctions(source, D) {
  const contest = D.contest, byId = new Map(D.entities.map((entity, index) => [entity.id, index]));
  return new Function("D", "contest", "byId", source + "\nreturn {sixNeighbors,nearest,select:typeof nearestRows==='function'?nearestRows:null};")(D, contest, byId);
}

function decodeLoop(data) {
  const binary = atob(data), bytes = new Uint8Array(binary.length);
  for (let i = 0; i < binary.length; i++) bytes[i] = binary.charCodeAt(i);
  return bytes;
}
const decodeCallback = data => Uint8Array.from(atob(data), character => character.charCodeAt(0));
async function unpack(bytes) {
  return new Response(new Blob([bytes]).stream().pipeThrough(new DecompressionStream("gzip"))).text();
}

function verifyEdgeCases(baselineSource, candidateSource) {
  const fixture = n => ({ entities: Array.from({ length: n }, (_, i) => ({ id: "x" + String(n - i).padStart(4, "0"),
    features: [1, 1, 1, 1, 1], v12_features: [1, 1, 1, 1, 1, 1], v12_neighbors: [] })), contest: { v12: { neighbors: 15 } } });
  let checks = 0;
  for (const n of [1, 2, 16, 32]) {
    const D = fixture(n), a = makeFunctions(baselineSource, D), b = makeFunctions(candidateSource, D);
    for (let index = 0; index < n; index++) {
      for (const excluded of [-1, 0, 1, 2, 3, 4, 5]) { assert.deepEqual(b.sixNeighbors(index, excluded), a.sixNeighbors(index, excluded)); checks++; }
      for (const excluded of [-1, 0, 1, 2, 3, 4]) { assert.deepEqual(b.nearest(index, excluded), a.nearest(index, excluded)); checks++; }
    }
    for (const count of [0, 1, 15, 100, undefined, -1]) {
      D.contest.v12.neighbors = count;
      assert.deepEqual(b.sixNeighbors(0, 0), a.sixNeighbors(0, 0)); checks++;
    }
    D.contest.v12.neighbors = 15;
    D.entities[0].features[0] = NaN; D.entities[0].v12_features[0] = NaN;
    assert.deepEqual(b.nearest(0, -1), a.nearest(0, -1)); checks++;
    assert.deepEqual(b.sixNeighbors(0, 1), a.sixNeighbors(0, 1)); checks++;
    delete D.entities[0].features; delete D.entities[0].v12_features;
    for (const key of ["nearest", "sixNeighbors"]) {
      const errors = [a, b].map(functions => { try { functions[key](0, 0); return null; } catch (error) { return { name: error.name, message: error.message }; } });
      assert.deepEqual(errors[1], errors[0]); checks++;
    }
  }
  const functions = makeFunctions(candidateSource, fixture(1)), compare = (a, b) => a.d - b.d;
  const tied = Array.from({ length: 40 }, (_, i) => ({ i, d: 1 }));
  assert.deepEqual(functions.select(tied, 15, compare), tied.slice().sort(compare).slice(0, 15)); checks++;
  assert.deepEqual(functions.select([], 15, compare), []); checks++;
  const empty = fixture(0), a = makeFunctions(baselineSource, empty), b = makeFunctions(candidateSource, empty);
  for (const key of ["nearest", "sixNeighbors"]) { assert.deepEqual(b[key](0, 0), a[key](0, 0)); checks++; }
  return checks;
}

async function measure(planFile) {
  const planBytes = fs.readFileSync(planFile), plan = JSON.parse(planBytes);
  assert.equal(process.version, plan.node); assert.equal(sha(fs.readFileSync(process.execPath)), plan.node_executable_sha256);
  for (const [file, record] of Object.entries(plan.frozen_inputs)) assert.equal(sha(fs.readFileSync(path.join(ROOT, file))), record.sha256, file);
  const rawLog = path.join(OUT, "paired-raw.jsonl");
  if (fs.existsSync(rawLog)) throw new Error("A raw timing log already exists; do not overwrite a run.");
  const log = value => fs.appendFileSync(rawLog, JSON.stringify(value) + "\n");
  const asset = fs.readFileSync(path.join(ROOT, plan.payload), "utf8");
  const match = asset.match(/window\.__SBERAI_ATLAS__=\{payload:([\s\S]+)\};\s*$/);
  assert.ok(match, "Actual classic-script payload missing");
  const packed = JSON.parse(match[1]), gzipBytes = decodeLoop(packed.data), text = await unpack(gzipBytes), D = JSON.parse(text);
  assert.equal(D.entities.length, 2016);
  const gzipSHA = sha(gzipBytes), textSHA = sha(text);
  log({ type: "environment", started_at_utc: new Date().toISOString(), plan_sha256: sha(planBytes), node: process.version,
    node_versions: process.versions, platform: process.platform, arch: process.arch, cpu: os.cpus()[0].model,
    node_argv: process.execArgv, env_threads: { UV_THREADPOOL_SIZE: process.env.UV_THREADPOOL_SIZE, OMP_NUM_THREADS: process.env.OMP_NUM_THREADS },
    process_priority: os.getPriority(), free_memory_bytes: os.freemem(), total_memory_bytes: os.totalmem(),
    fromBase64: typeof Uint8Array.fromBase64, payload_sha256: sha(Buffer.from(asset)), gzip_sha256: gzipSHA,
    gzip_bytes: gzipBytes.length, uncompressed_sha256: textSHA, uncompressed_utf8_bytes: Buffer.byteLength(text) });
  const baselineSource = fs.readFileSync(path.join(OUT, "baseline-functions.js"), "utf8");
  const candidateSource = fs.readFileSync(path.join(OUT, "candidate-functions.js"), "utf8");
  const a = makeFunctions(baselineSource, D), b = makeFunctions(candidateSource, D);
  const edgeChecks = verifyEdgeCases(baselineSource, candidateSource);
  let fullChecks = 0, selfIncluded = 0;
  for (let index = 0; index < D.entities.length; index++) {
    for (let excluded = -1; excluded <= 5; excluded++) {
      const before = a.sixNeighbors(index, excluded), after = b.sixNeighbors(index, excluded);
      assert.deepEqual(after, before); assert.ok(after.every(row => row.i !== index)); fullChecks++;
    }
    for (let excluded = -1; excluded <= 4; excluded++) {
      const before = a.nearest(index, excluded), after = b.nearest(index, excluded);
      assert.deepEqual(after, before); assert.ok(after.every(row => row.i !== index)); fullChecks++;
    }
    if (index % 100 === 0) await new Promise(resolve => setImmediate(resolve));
  }
  log({ type: "correctness", status: "PASS", exhaustive_deep_equal_checks: fullChecks, synthetic_checks: edgeChecks,
    distance_comparison: "Exact JS numbers with DeepEqual; no tolerance or rounding", self_included: selfIncluded });

  const scenario = (functions, key) => {
    const start = performance.now(); let checksum = 0;
    for (let query = 0; query < plan.scenario.queries_per_batch; query++) {
      const index = query * 97 % D.entities.length, excluded = query % (key === "sixNeighbors" ? 6 : 5);
      for (const row of functions[key](index, excluded)) checksum += row.i + row.d;
    }
    return { ms: performance.now() - start, checksum };
  };
  const scenarioRows = {};
  for (const key of plan.scenario.functions) {
    assert.equal(scenario(a, key).checksum, scenario(b, key).checksum);
    scenarioRows[key] = [];
    for (let pair = 0; pair < plan.repeats; pair++) {
      const order = plan.pair_order[pair], row = { type: "scenario", function: key, pair, order, queries: plan.scenario.queries_per_batch };
      for (const variant of order) row[variant] = scenario(variant === "A" ? a : b, key);
      assert.equal(row.A.checksum, row.B.checksum);
      log(row); scenarioRows[key].push(row);
      await new Promise(resolve => setImmediate(resolve));
    }
  }

  async function timedDecode(variant) {
    const start = performance.now(), bytes = (variant === "A" ? decodeCallback : decodeLoop)(packed.data), base64End = performance.now();
    const decoded = await unpack(bytes), gzipEnd = performance.now(), object = JSON.parse(decoded), parseEnd = performance.now();
    assert.equal(sha(bytes), gzipSHA); assert.equal(sha(decoded), textSHA); assert.deepEqual(object, D);
    return { base64_ms: base64End - start, gzip_utf8_ms: gzipEnd - base64End, json_parse_ms: parseEnd - gzipEnd,
      total_ms: parseEnd - start, gzip_sha256: sha(bytes), text_sha256: sha(decoded) };
  }
  await timedDecode("A"); await timedDecode("B");
  const decodeRows = [];
  for (let pair = 0; pair < plan.repeats; pair++) {
    const order = plan.pair_order[pair], row = { type: "decode", pair, order };
    for (const variant of order) row[variant] = await timedDecode(variant);
    log(row); decodeRows.push(row);
  }
  const summarize = (rows, get) => ({
    A_median_ms: median(rows.map(row => get(row.A))), B_median_ms: median(rows.map(row => get(row.B))),
    A_range_ms: [Math.min(...rows.map(row => get(row.A))), Math.max(...rows.map(row => get(row.A)))],
    B_range_ms: [Math.min(...rows.map(row => get(row.B))), Math.max(...rows.map(row => get(row.B)))],
    paired_relative_reduction_median: median(rows.map(row => 1 - get(row.B) / get(row.A))),
    pairs_improved: rows.filter(row => get(row.B) < get(row.A)).length, pairs: rows.length,
  });
  const summary = { status: "PASS", completed_at_utc: new Date().toISOString(), plan_sha256: sha(planBytes),
    raw_sha256: sha(fs.readFileSync(rawLog)), payload_sha256: plan.frozen_inputs[plan.payload].sha256,
    node: process.version, process_memory_usage_bytes: process.memoryUsage(),
    correctness: { exhaustive_checks: fullChecks, synthetic_checks: edgeChecks, exact_distances: true, self_exclusion: true },
    scenario: Object.fromEntries(Object.entries(scenarioRows).map(([key, rows]) => [key, summarize(rows, result => result.ms)])),
    decode: Object.fromEntries(["base64_ms", "gzip_utf8_ms", "json_parse_ms", "total_ms"].map(key => [key, summarize(decodeRows, result => result[key])])),
    integration_rule_passed: Object.values(scenarioRows).every(rows => rows.every(row => row.B.ms < row.A.ms)),
    html_bytes: { baseline: fs.statSync(path.join(OUT, "baseline-template.html")).size, meaning: "Source template, not transferred HTML. Deployed HTML sizes are recorded after integration." },
    unknown: plan.unknown };
  saveJSON(path.join(OUT, "summary.json"), summary);
  console.log(JSON.stringify(summary));
}

function deployment(templateFile) {
  const template = fs.readFileSync(templateFile, "utf8"), candidate = fs.readFileSync(path.join(OUT, "candidate-functions.js"), "utf8");
  for (const name of ["sixNeighbors", "nearest"]) assert.equal(functionLine(template, name), functionLine(candidate, name));
  assert.ok(template.includes(SELECT), "The deployed selection helper differs from the measured helper.");
  const result = { status: "PASS", checked_at_utc: new Date().toISOString(), template: relative(templateFile),
    template_sha256: sha(Buffer.from(template)), candidate_functions_sha256: sha(Buffer.from(candidate)) };
  saveJSON(path.join(OUT, "integration.json"), result); console.log(JSON.stringify(result));
}

(async () => {
  const [command, argument] = process.argv.slice(2);
  if (command === "prepare") prepare();
  else if (command === "measure") await measure(path.resolve(ROOT, argument || relative(path.join(OUT, "plan.json"))));
  else if (command === "deployment") deployment(path.resolve(ROOT, argument || "web/research_template.html"));
  else throw new Error("Usage: competition_performance.cjs prepare|measure [plan]|deployment [template]");
})().catch(error => { console.error(error.stack); process.exitCode = 1; });
