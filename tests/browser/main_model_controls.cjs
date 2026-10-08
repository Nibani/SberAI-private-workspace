const fs = require('node:fs');
const path = require('node:path');
const vm = require('node:vm');
const assert = require('node:assert/strict');
const root = path.resolve(process.argv[2] || '.');
const template = fs.readFileSync(path.join(root, 'web/research_template.html'), 'utf8');
function block(from, to) {
  const start = template.indexOf(from), end = template.indexOf(to, start);
  assert(start >= 0 && end > start, `Controller boundaries exist: ${from}`);
  return template.slice(start, end);
}
const source = block('const workspaceViews=', 'function el(tag,text,cls)') +
  block('function syncMapControls(){', 'function renderContest(){') +
  block('function syncNeighborControls(){', 'function renderIdentity(){');
const calls = [], animations = [];
class Target {
  constructor(id) { this.id = id; this.dataset = {}; this.attrs = {}; this.handlers = {}; this.hidden = false; this.inert = false; this.value = ''; this.textContent = ''; this.active = []; this.style = {setProperty: (key, value) => this.attrs[key] = value}; }
  setAttribute(key, value) { this.attrs[key] = value; }
  addEventListener(key, fn) { this.handlers[key] = fn; }
  focus() { document.activeElement = this; }
  getAnimations() { return this.active.filter(a => !a.cancelled); }
  animate(frames, options) {
    const animation = {target: this.id, frames, options, cancelled: false, finished: {then: fn => animation.finish = fn}, cancel: () => animation.cancelled = true};
    this.active.push(animation); animations.push(animation); return animation;
  }
}
const nodes = Object.fromEntries(['method','map-model','map-year','method-note','quick','search','exclude','territory-inspector','neighbor-results'].map(id => [id, new Target(id)]));
const panels = ['map','analogs','dynamics'].map(view => { const n = new Target('workspace-' + view); n.dataset.workspacePanel = view; nodes[n.id] = n; return n; });
const tabs = ['map','analogs','dynamics'].map(view => { const n = new Target('view-' + view); n.dataset.view = view; nodes[n.id] = n; return n; });
const tabbar = new Target('tabs');
const document = {body: {dataset: {}}, activeElement: null, handlers: {},
  addEventListener(key, fn) { this.handlers[key] = fn; },
  querySelector(selector) { return selector === '.workspace-tabs' ? tabbar : null; },
  querySelectorAll(selector) { return selector === '[data-view]' ? tabs : selector === '[data-workspace-panel]' ? panels : []; }};
let reduced = false;
const preference = {get matches() { return reduced; }, addEventListener(key, fn) { this.listener = fn; }};
const context = {console, Map, document, getComputedStyle: () => ({opacity: '.93', transform: 'matrix(1, 0, 0, 1, 0, 2)'}),
  matchMedia: () => preference, state: {index: 0, method: 'v12', mapModel: 'v12_types', mapYear: '2024', mapMode: 'relative', exclude: 2, view: 'map'},
  contest: {v12: {}, map_models: {}}, D: {entities: [{id: 'one'}, {id: 'two'}]}, $: id => nodes[id], label: e => e.id, baseNeighbors: () => []};
for (const name of ['renderIdentity','renderProfile','renderScatter','renderNeighbors','renderTimeline','renderEgo','renderMap','renderMapLegend']) context[name] = () => calls.push(name);
context.renderIdentity = () => { nodes['territory-inspector'].textContent = context.D.entities[context.state.index].id; calls.push('renderIdentity'); };
vm.createContext(context); vm.runInContext(source, context);
const run = code => vm.runInContext(code, context), cases = [];
run('choose(0)');
assert.equal(nodes.method.value, 'v12'); assert.equal(nodes.quick.value, '0');
assert.equal(animations.length, 0, 'Initial facts appear without animation'); cases.push('main defaults and immediate initial facts');
for (const method of ['profile','consensus','transport']) { run(`selectPeerMethod('${method}')`); assert.equal(context.state.method, 'v12'); }
assert.equal(nodes.exclude.value, '-1'); cases.push('historical peer modes cannot replace main model');
run("selectMapModel('frozen'); selectMapModel('other')");
assert.equal(context.state.mapModel, 'v12_types'); assert.equal(nodes['map-model'].value, 'v12_types'); cases.push('historical map cannot replace joint model');
context.contest.v12 = null; run("selectPeerMethod('transport'); selectMapModel('other')");
assert.equal(context.state.method, 'profile'); assert.equal(context.state.mapModel, 'frozen'); cases.push('legacy payload remains usable');
context.contest.v12 = {}; context.state.mapModel = 'v12_types';
Object.assign(context.state, {index: 1, mapYear: '2024', mapMode: 'relative', exclude: 3});
const retained = JSON.stringify([context.state.index,context.state.mapYear,context.state.mapMode,context.state.exclude]);
run("setWorkspaceView('analogs'); setWorkspaceView('dynamics'); setWorkspaceView('map')");
assert.equal(JSON.stringify([context.state.index,context.state.mapYear,context.state.mapMode,context.state.exclude]), retained);
assert.equal(run("setWorkspaceView('invalid')"), false); cases.push('three views preserve territory, year, correction and category');
run('setupWorkspace()'); document.handlers.keydown();
let prevented = false; tabs[0].handlers.keydown({key: 'ArrowRight', preventDefault: () => prevented = true});
assert(prevented); assert.equal(context.state.view, 'analogs'); assert.equal(document.activeElement.id, 'view-analogs');
assert.equal(tabs[1].attrs['aria-selected'], 'true'); assert.equal(tabs[0].tabIndex, -1); assert.equal(animations.length, 0); cases.push('keyboard tabs move focus immediately without animation');
document.handlers.pointerdown(); run('choose(0); choose(1)');
assert.equal(nodes['territory-inspector'].textContent, 'two', 'Facts update before motion finishes');
const [first, second] = animations.slice(-2); assert(first.cancelled); assert.equal(second.frames[0].opacity, '.93');
assert.equal(second.frames[0].transform, 'matrix(1, 0, 0, 1, 0, 2)'); assert.equal(second.options.duration, 200); cases.push('selection retargets current presentation without delaying facts');
run("setWorkspaceView('map'); setWorkspaceView('dynamics'); setWorkspaceView('analogs')");
assert.equal(context.state.view, 'analogs'); assert.equal(nodes['workspace-analogs'].hidden, false); assert.equal(nodes['workspace-map'].inert, true);
for (const animation of animations) if (animation.cancelled && animation.finish) animation.finish();
assert.equal(nodes['workspace-analogs'].hidden, false, 'Stale completion cannot hide the latest view'); cases.push('rapid view switches cancel stale transitions');
reduced = true; preference.listener(); const before = animations.length;
run("choose(0); setWorkspaceView('map'); animatePanel('neighbor-results',180)");
assert.equal(animations.length, before); assert.equal(nodes['territory-inspector'].textContent, 'one');
assert.deepEqual(panels.map(n => n.hidden), [false,true,true]); cases.push('reduced motion cancels existing effects and updates state immediately');
const result = {status: 'PASS', scope: 'Extracted controller functions with deterministic Node DOM and animation mocks; no browser or pixels', cases, calls: calls.length, animations: animations.length};
if (process.argv[3]) fs.writeFileSync(path.resolve(process.argv[3]), JSON.stringify(result, null, 2) + '\n');
console.log(JSON.stringify(result));
