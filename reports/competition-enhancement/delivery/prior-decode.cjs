const fs = require('node:fs');
const vm = require('node:vm');
const assert = require('node:assert/strict');
const {performance} = require('node:perf_hooks');
const {createHash} = require('node:crypto');

const html = fs.readFileSync('docs/index.html', 'utf8');
const asset = 'docs/' + html.match(/src="(assets\/payload-[a-f0-9]+\.js)"/)[1];
const context = {window: {}};
vm.runInNewContext(fs.readFileSync(asset, 'utf8'), context);
const packed = context.window.__SBERAI_ATLAS__.payload;
function before(data) {return Uint8Array.from(atob(data), c => c.charCodeAt(0));}
function after(data) {
  const binary = atob(data), bytes = new Uint8Array(binary.length);
  for (let i = 0; i < binary.length; i++) bytes[i] = binary.charCodeAt(i);
  return bytes;
}
async function unpack(bytes) {
  return JSON.parse(await new Response(new Blob([bytes]).stream().pipeThrough(new DecompressionStream('gzip'))).text());
}
function summary(samples) {
  const sorted = [...samples].sort((a,b)=>a-b), middle = Math.floor(sorted.length/2);
  return {samples_ms: samples, median_ms: sorted.length%2 ? sorted[middle] : (sorted[middle-1]+sorted[middle])/2};
}
(async () => {
  const original = before(packed.data), changed = after(packed.data);
  assert.deepStrictEqual(changed, original);
  const payload = await unpack(original);
  assert.deepStrictEqual(await unpack(changed), payload);
  for (const data of ['', 'AAECA/7/', packed.data.slice(0, 400)]) assert.deepStrictEqual(after(data), before(data));
  const decode = {before: [], after: []}, full = {before: [], after: []};
  for (let r = 0; r < 7; r++) {
    for (const name of r%2 ? ['after', 'before'] : ['before', 'after']) {
      const fn = name === 'before' ? before : after;
      let t = performance.now(); const bytes = fn(packed.data); const elapsed = performance.now()-t;
      const decoded = await unpack(bytes); const whole = performance.now()-t;
      assert.deepStrictEqual(decoded, payload);
      if (r > 0) {decode[name].push(elapsed);full[name].push(whole);}
    }
  }
  console.log(JSON.stringify({status: 'PASS', node: process.version, asset, asset_bytes: fs.statSync(asset).size,
    compressed_bytes: original.length, decoded_sha256: createHash('sha256').update(JSON.stringify(payload)).digest('hex'),
    exact_bytes: true, all_payload_fields_deep_equal: true, decode: Object.fromEntries(Object.entries(decode).map(([k,v])=>[k,summary(v)])),
    decode_decompress_parse: Object.fromEntries(Object.entries(full).map(([k,v])=>[k,summary(v)])), browser: false}, null, 2));
})().catch(error=>{console.error(error);process.exitCode=1;});
