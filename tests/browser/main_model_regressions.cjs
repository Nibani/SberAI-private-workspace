// Run only in the authorized remote runner; this entry point never opens file URLs.
if (process.env.GITHUB_ACTIONS !== 'true') throw new Error('V3 browser review requires the authorized GitHub Actions runner');
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const crypto = require('node:crypto');
const zlib = require('node:zlib');
const checkMapWorkspace = require('./map_workspace_checks.cjs');
const {chromium} = require(process.env.PLAYWRIGHT_PATH || 'playwright');
const base = new URL(process.env.TEST_BASE_URL || '');
assert(['http:','https:'].includes(base.protocol), 'TEST_BASE_URL must be HTTP or HTTPS');
if (base.protocol === 'http:') assert(['127.0.0.1','localhost','[::1]'].includes(base.hostname), 'HTTP is limited to the remote runner loopback');
const file = path.resolve(process.argv[2] || 'docs/index.html');
const output = path.resolve(process.argv[3] || 'results/browser-v3');
fs.mkdirSync(output, {recursive: true});
const hash = value => crypto.createHash('sha256').update(value).digest('hex');
const html = fs.readFileSync(file, 'utf8');
const source = html.match(/<script src="(assets\/payload-[a-f0-9]{64}\.js)"><\/script>/);
const script = source ? fs.readFileSync(path.join(path.dirname(file), source[1]), 'utf8') : html;
const packed = script.match(/\{"encoding":"gzip-base64","data":"([A-Za-z0-9+/=]+)"\}/);
assert(packed, 'Lossless atlas data package exists');
const data = JSON.parse(zlib.gunzipSync(Buffer.from(packed[1], 'base64')));
const report = {status: 'RUNNING', scope: 'Real Chromium on the authorized GitHub runner over HTTP; no local browser', base: base.href,
  htmlSha256: hash(Buffer.from(html)), dataAssetSha256: hash(Buffer.from(script)), geometrySha256: hash(JSON.stringify(data.contest.map)), checks: []};
const save = () => fs.writeFileSync(path.join(output, 'result.json'), JSON.stringify(report, null, 2) + '\n');
save();
const expectedNeighbors = (index, excluded) => {
  if (excluded < 0) return data.entities[index].v12_neighbors;
  const a = data.entities[index];
  return data.entities.map((b,i) => ({i, d: i === index ? Infinity : Math.sqrt(a.v12_features.reduce((sum,v,j) => sum + (j === excluded ? 0 : (v-b.v12_features[j])**2), 0))}))
    .sort((a,b) => a.d-b.d || a.i-b.i).slice(0,15).map(x => data.entities[x.i].id);
};
async function observePrimaryRoute(page) {
  return page.evaluate(()=>{
    const start=document.querySelector('.hero'),end=document.getElementById('main-limitations');
    const startY=start.getBoundingClientRect().top+scrollY,endY=end.getBoundingClientRect().bottom+scrollY;
    const inspect=node=>{
      const r=node.getBoundingClientRect(),style=getComputedStyle(node);
      const rendered=r.width>1&&r.height>1&&style.visibility!=='hidden'&&style.display!=='none'&&Number(style.opacity)>0&&!node.closest('.sr,[hidden],[inert]');
      const inViewport=rendered&&r.bottom>0&&r.top<innerHeight&&r.right>0&&r.left<innerWidth;
      const label=node.getAttribute('aria-label')||node.labels?.[0]?.textContent||node.selectedOptions?.[0]?.textContent||node.textContent;
      return{id:node.id||null,tag:node.tagName,label:String(label||'').replace(/\s+/g,' ').trim(),value:node.value??null,selectedLabel:node.selectedOptions?.[0]?.textContent?.trim()??null,
        rendered,inViewport,fullyInViewport:inViewport&&r.top>=0&&r.left>=0&&r.bottom<=innerHeight&&r.right<=innerWidth,
        rect:{x:r.x,y:r.y,width:r.width,height:r.height,documentTop:r.top+scrollY,documentBottom:r.bottom+scrollY}};
    };
    const controls=[...document.querySelectorAll('.hero a,.atlas-workspace input,.atlas-workspace select,.atlas-workspace button,.atlas-workspace summary')].map(inspect).filter(item=>item.rendered);
    const essentials=['h1','#finding-title','.proof-main','.proof-caveat','#territory-title','#search','#quick','.workspace-tabs','#territory-map','#territory-inspector','#read-title','#main-limitations'].map(selector=>({selector,...inspect(document.querySelector(selector))}));
    return{url:location.href,ready:window.__ATLAS_READY__===true,fonts:document.fonts.status,epochMs:Date.now(),scrollX,scrollY,
      viewport:{width:innerWidth,height:innerHeight,devicePixelRatio},
      activeView:document.querySelector('[role="tab"][aria-selected="true"]').id,
      primaryRoute:{start:'.hero',end:'#main-limitations',startY,endY,cssPx:endY-startY,viewportHeights:(endY-startY)/innerHeight},
      controls,controlsInFirstScreen:controls.filter(item=>item.inViewport),essentials};
  });
}
async function checkLayout(page, label) {
  const issues = await page.evaluate(() => {
    const issues = [], visible = node => { const r=node.getBoundingClientRect(); return r.width>0&&r.height>0; };
    if (document.documentElement.scrollWidth>innerWidth+1) issues.push({kind:'page-overflow',width:document.documentElement.scrollWidth});
    const targets='input:not([type="checkbox"]):not([type="range"]),select,button:not(.table-link),[role="tab"],.check,.map-options > summary,.proof > summary';
    const coarseTargets=matchMedia('(pointer: coarse)').matches?',button.table-link,.example-details > summary':'';
    for (const node of document.querySelectorAll(targets+coarseTargets)) {
      if (!visible(node)) continue; const r=node.getBoundingClientRect();
      if(r.height<43.5)issues.push({kind:'small-control',id:node.id,height:r.height});
      if(!node.closest('.table-wrap,.table-scroll,.chart-scroll')&&(r.left<-.5||r.right>innerWidth+.5))issues.push({kind:'control-clipping',id:node.id,left:r.left,right:r.right});
    }
    for (const node of document.querySelectorAll('h1,h2,h3,label,.identity,.fact strong,.proof-main strong,.bar-label,[role="tab"],.proof > summary')) {
      if(!visible(node)||node.closest('.table-wrap,.table-scroll,.chart-scroll'))continue;
      if(node.scrollWidth>node.clientWidth+1||node.scrollHeight>node.clientHeight+1)issues.push({kind:'text-clipping',id:node.id,text:node.textContent.slice(0,100)});
    }
    for(const id of ['map-model','method']) {const node=document.getElementById(id);if(!visible(node))continue;const style=getComputedStyle(node),canvas=document.createElement('canvas'),context=canvas.getContext('2d');context.font=`${style.fontWeight} ${style.fontSize} ${style.fontFamily}`;
      if(context.measureText(node.selectedOptions[0].textContent).width>node.clientWidth-parseFloat(style.paddingLeft)-parseFloat(style.paddingRight)-24)issues.push({kind:'select-label-clipping',id});
    }
    return issues;
  });
  assert.deepEqual(issues, [], `${label}: controls and text fit, targets >=44px, no whole-page overflow`);
}
async function checkTerritory(page, excluded=-1) {
  const index=+await page.locator('#quick').inputValue(),entity=data.entities[index];
  assert((await page.locator('#identity').textContent()).includes(entity.name));
  assert((await page.locator('#identity').textContent()).includes(data.contest.v12.types[entity.v12_type].name));
  assert.deepEqual(await page.locator('#neighbors tr').evaluateAll(rows=>rows.map(row=>row.dataset.entityId)), expectedNeighbors(index,excluded));
  const paths=await page.locator('#territory-map path.map-shape').evaluateAll(nodes=>nodes.map(node=>node.getAttribute('d')));
  assert.deepEqual(paths,data.contest.map.paths.map(p=>p.d),'Every map path is preserved in original order');
  const selected=page.locator('#territory-map path.map-shape.selected');
  assert.equal(await selected.count(),1);
  assert.equal(await page.locator('#territory-map .map-outline-selected .map-outline-stroke').getAttribute('d'),await selected.getAttribute('d'));
  assert.equal(await page.locator('#map-model').inputValue(),'v12_types');
  assert.equal(await page.locator('#method').inputValue(),'v12');
  return index;
}
async function retainedState(page) {
  return page.evaluate(()=>Object.fromEntries(['quick','map-year','map-mode','exclude','month','series'].map(id=>[id,document.getElementById(id).value])));
}
async function runViewports(browser) {
  for(const viewport of [{width:1440,height:900},{width:390,height:844}]) {
    const mobile=viewport.width<600;
    const record={viewport,mobileEmulation:mobile,errors:[],pageErrors:[],consoleErrors:[],httpErrors:[],failedRequests:[],blockedRequests:[],actions:[],screenshots:[]};report.checks.push(record);save();
    const context=await browser.newContext({viewport,reducedMotion:'no-preference',hasTouch:mobile,isMobile:mobile});
    await context.tracing.start({screenshots:true,snapshots:true,sources:true});
    await context.route('**/*',route=>{if(new URL(route.request().url()).origin===base.origin)return route.continue();record.blockedRequests.push({url:route.request().url(),reason:'outside authorized origin'});return route.abort()});
    const page=await context.newPage();
    const activate=locator=>mobile?locator.tap():locator.click();
    record.userPath={goal:'search territory → 15 analogs → exclude category → selected month',unit:'semantic input operation; not assertion count or physical tap count',status:'NOT_STARTED',entries:[],attempted:0,completed:0};
    const userAction=async(stage,input,target,operation)=>{
      const entry={number:record.userPath.entries.length+1,stage,input,target,status:'RUNNING',startedUtc:new Date().toISOString()};
      record.userPath.entries.push(entry);record.userPath.status='RUNNING';record.userPath.attempted=record.userPath.entries.length;save();
      try{await operation();entry.status='COMPLETED'}catch(error){entry.status='FAILED';entry.error=String(error);throw error}
      finally{entry.finishedUtc=new Date().toISOString();record.userPath.completed=record.userPath.entries.filter(item=>item.status==='COMPLETED').length;save()}
    };
    page.on('pageerror',error=>{record.pageErrors.push(error.message);record.errors.push(error.message)});
    page.on('console',message=>{if(message.type()==='error'){record.consoleErrors.push({text:message.text(),location:message.location()});record.errors.push(message.text())}});
    page.on('response',response=>{if(response.status()>=400)record.httpErrors.push({url:response.url(),status:response.status(),statusText:response.statusText()})});
    page.on('requestfailed',request=>record.failedRequests.push({url:request.url(),method:request.method(),error:request.failure()?.errorText}));
    await page.addInitScript(()=>{
      window.__MOTION_REVIEW__=[];const animate=Element.prototype.animate,cancel=Animation.prototype.cancel;
      Element.prototype.animate=function(frames,options){const result=animate.call(this,frames,options);result.__reviewTarget=this.id;window.__MOTION_REVIEW__.push({kind:'start',id:this.id,frames,options});return result};
      Animation.prototype.cancel=function(){window.__MOTION_REVIEW__.push({kind:'cancel',id:this.__reviewTarget});return cancel.call(this)};
    });
    try {
      await page.goto(base.href,{waitUntil:'load'});await page.waitForFunction(()=>window.__ATLAS_READY__===true);await page.evaluate(()=>document.fonts.ready);
      await context.tracing.group(`first-screen-${viewport.width}`);
      try{
        const observation=await observePrimaryRoute(page),name=`first-screen-${viewport.width}.png`;
        record.firstScreen={...observation,screenshot:name,htmlSha256:report.htmlSha256,dataAssetSha256:report.dataAssetSha256,userPathActionsBeforeCapture:record.userPath.attempted};
        const pixels=await page.screenshot({path:path.join(output,name),fullPage:false,animations:'allow'});record.screenshots.push(name);
        record.firstScreen.screenshotSha256=hash(pixels);record.firstScreen.scrollAfterCapture=await page.evaluate(()=>({x:scrollX,y:scrollY}));save();
        assert.equal(observation.ready,true);assert.equal(observation.fonts,'loaded');assert.equal(observation.scrollY,0);assert.equal(observation.scrollX,0);
        assert.deepEqual(record.firstScreen.scrollAfterCapture,{x:0,y:0});assert.equal(record.firstScreen.userPathActionsBeforeCapture,0);
      }finally{await context.tracing.groupEnd()}

      const links=await page.locator('a[href]').evaluateAll(anchors=>anchors.map(a=>({href:a.getAttribute('href'),resolved:a.href})).filter(link=>link.href&&!link.href.startsWith('#')));
      assert.equal(links.filter(link=>link.href.startsWith('https://github.com/Nibani/SberAI/blob/main/')).length,0,'Included scientific files use the current kit paths');
      record.includedLocalLinks=[];
      for(const href of new Set(links.map(link=>link.resolved).filter(href=>new URL(href).origin===base.origin))){
        const url=new URL(href);url.hash='';
        try{const response=await context.request.get(url.href);record.includedLocalLinks.push({url:url.href,status:response.status()});if(!response.ok())record.httpErrors.push({url:url.href,status:response.status(),scope:'included-file-link'});assert(response.ok(),`Included file link responds: ${url.href}`)}
        catch(error){record.failedRequests.push({url:url.href,error:String(error),scope:'included-file-link'});throw error}
      }
      record.pointerCapabilities=await page.evaluate(()=>({coarse:matchMedia('(pointer: coarse)').matches,hover:matchMedia('(hover: hover)').matches,touchPoints:navigator.maxTouchPoints}));
      if(mobile){assert.equal(record.pointerCapabilities.coarse,true);assert.equal(record.pointerCapabilities.hover,false);assert(record.pointerCapabilities.touchPoints>0)}
      await checkLayout(page,'initial');
      assert.equal(await page.locator('h1').textContent(),'Экономические соседи');
      for(const heading of await page.locator('h1,h2,h3').all()) {const style=await heading.evaluate(n=>{const s=getComputedStyle(n);return [s.fontFamily,s.fontWeight,s.fontSynthesis]});assert(style[0].includes('Cinzel RU'));assert.equal(style[1],'400');assert.equal(style[2],'none')}
      assert((await page.locator('#finding-title').textContent()).includes('Ошибка роста расходов 2024'));
      assert((await page.locator('.proof-caveat').textContent()).includes('ретроспективная'));
      assert((await page.locator('.hero-proof').textContent()).includes('2,486'));
      assert.equal(await page.locator('.hero-proof').evaluate(n=>getComputedStyle(n).opacity),'1');
      assert.equal(await page.locator('.proof-main').evaluate(n=>getComputedStyle(n).backgroundColor),'rgba(0, 0, 0, 0)','MAE belongs to the shared text ribbon');
      assert.equal(await page.locator('#map-model').isVisible(),false);assert.equal(await page.locator('#method').isVisible(),false,'Single-option rules do not promise a choice');
      if(!mobile){const first=await page.locator('#territory-map').boundingBox();assert(first.y<viewport.height-180,'A useful part of the map is present on the first screen');const bars=await page.locator('#profile').boundingBox();assert(bars.y<viewport.height-120,'The inspector profile is visible on the first screen')}
      assert.equal(await page.locator('#archive-v11').getAttribute('open'),null);
      assert.equal(await page.locator('#main-appendices').getAttribute('open'),null);
      assert.equal(await page.locator('#archive-map-model,#archive-method').count(),0,'Comparisons cannot substitute the main model');
      const visibleVersions=await page.evaluate(()=>/v1\.[12]/i.test(document.body.innerText.replace(/(?:reports|data|scripts)[^\s]*/g,'')));
      assert.equal(visibleVersions,false,'The interface names models by meaning');
      await checkTerritory(page);record.actions.push('initial finding, main model and full geometry');
      await userAction('selection',mobile?'tap':'click','hero-search',()=>activate(page.locator('.hero-links [data-focus-search]')));await page.waitForFunction(()=>document.activeElement.id==='search');
      const settleAnchor=()=>page.evaluate(()=>new Promise(resolve=>requestAnimationFrame(()=>requestAnimationFrame(resolve))));
      await settleAnchor();assert.equal(await page.evaluate(()=>document.activeElement.id),'search','Hash navigation does not steal search focus');
      await page.goBack();await page.waitForURL(url=>url.hash!=='#territory-title');await page.goForward();await page.waitForURL(url=>url.hash==='#territory-title');await settleAnchor();assert.equal(await page.evaluate(()=>document.activeElement.id),'search');
      const unchanged=await page.locator('#quick').inputValue();await page.locator('#search').fill('несуществующая территория 987654321');await page.locator('#search').press('Enter');assert.equal(await page.locator('#quick').inputValue(),unchanged);assert((await page.locator('#search-feedback').textContent()).includes('Совпадений нет'));
      await page.locator('#search').fill(data.entities[0].region);await page.locator('#search').press('Enter');assert.equal(await page.locator('#quick').inputValue(),unchanged);assert((await page.locator('#search-feedback').textContent()).includes('Уточните'));
      await page.locator('#search').fill(' ');await page.locator('#search').press('Enter');assert.equal(await page.locator('#quick').inputValue(),unchanged);assert((await page.locator('#search-feedback').textContent()).includes('Введите'));record.actions.push('stable anchor focus, browser Back/Forward and explicit ambiguous, empty and missing searches');
      const target=data.entities[250];await userAction('selection','fill','#search',()=>page.locator('#search').fill(target.name));await userAction('selection','press Enter','#search',()=>page.locator('#search').press('Enter'));
      assert((await page.locator('#identity').textContent()).includes(target.name));await checkTerritory(page);record.actions.push('search and immediate territory card');
      await userAction('analogs',mobile?'tap':'click','#view-analogs',()=>activate(page.locator('#view-analogs')));await userAction('analogs','native selectOption 1','#exclude',()=>page.locator('#exclude').selectOption('1'));await checkTerritory(page,1);
      assert((await page.locator('#preserved').textContent()).includes('из 15'));await checkLayout(page,'analogs');record.actions.push('exact 15NN and excluded-category ordering');
      await userAction('dynamics',mobile?'tap':'click','#view-dynamics',()=>activate(page.locator('#view-dynamics')));await userAction('dynamics','native selectOption 5','#month',()=>page.locator('#month').selectOption('5'));await page.locator('#series').selectOption('total');
      const index=+await page.locator('#quick').inputValue(),peers=expectedNeighbors(index,1).map(id=>data.entities.find(e=>e.id===id));
      const median=peers.map(e=>e.totals[5]).sort((a,b)=>a-b)[7],fmt=new Intl.NumberFormat('ru-RU',{maximumFractionDigits:2});
      const month=await page.locator('#month-summary').textContent();assert(month.includes(data.periods[5].slice(0,7)));assert(month.includes(fmt.format(data.entities[index].totals[5])));assert(month.includes(fmt.format(median)));
      assert.equal(await page.locator('#monthly-membership .cluster-strip span').count(),24);await checkLayout(page,'dynamics');record.actions.push('month values, peer median and monthly groups');
      record.userPath.status='COMPLETE';record.userPath.finalState=await retainedState(page);
      record.userPath.routeAtGoal=await observePrimaryRoute(page);record.userPath.byStage=Object.fromEntries(['selection','analogs','dynamics'].map(stage=>[stage,record.userPath.entries.filter(item=>item.stage===stage&&item.status==='COMPLETED').length]));save();
      await activate(page.locator('#view-map'));await page.locator('#map-year').selectOption('2024');await activate(page.locator('.map-options > summary'));
      await page.locator('#map-transitions').check();for(const mode of ['raw','relative']){await page.locator('#map-mode').selectOption(mode);assert.equal(await page.locator('#territory-map path.map-shape.changed').count(),data.contest.map_models.v12_types.dynamics[mode].changed)}
      const preserved=await retainedState(page);await activate(page.locator('.hero-links a[href="#evidence-title"]'));await activate(page.locator('#main-appendices > summary'));
      assert.equal(await page.locator('#profile-cards .v12-type').count(),4);assert(await page.locator('#v12-types-title').isVisible());
      await activate(page.locator('#proof-model > summary'));await activate(page.locator('#archive-v11 > summary'));assert(await page.locator('#network-current-title').isVisible());
      assert((await page.locator('#comparison-models').textContent()).includes('KMeans4'));
      for(const id of ['proof-analogs','proof-dynamics','proof-added-value','proof-sources'])await activate(page.locator(`#${id} > summary`));
      for(const id of ['v12-types-title','v12-network-title','v12-methods-title','v12-dynamics-title','v12-practice-title']){assert.equal(await page.locator(`#${id}`).count(),1);assert(await page.locator(`#${id}`).isVisible());assert(await page.locator(`a[href="#${id}"]`).count()>=1)}
      const chartLabels=await page.locator('.v12-svg text').evaluateAll(nodes=>nodes.map(n=>{const b=n.getBoundingClientRect(),v=n.ownerSVGElement.getBoundingClientRect();return{text:n.textContent,x:b.left-v.left,end:b.right-v.left,width:v.width}}).filter(r=>r.x<-.5||r.end>r.width+.5));
      assert.deepEqual(chartLabels,[],'Saved chart labels fit after their SVG transforms');
      // Every saved scientific table remains available after presentation regrouping.
      const tableRetention=await page.evaluate(fragment=>{const source=document.createElement('div');source.innerHTML=fragment;const text=n=>n.textContent.replace(/\s+/g,' ').trim();const available=new Map();for(const table of document.querySelectorAll('table')){const key=text(table);available.set(key,(available.get(key)||0)+1)}return [...source.querySelectorAll('table')].map(text).filter(value=>{const count=available.get(value)||0;if(count)available.set(value,count-1);return !count})},data.contest.v12.ui_fragment);
      assert.deepEqual(tableRetention,[],'All saved scientific tables remain in the DOM');
      const paragraphRetention=await page.evaluate(fragment=>{const source=document.createElement('div');source.innerHTML=fragment;const text=n=>n.textContent.replace(/Архив v1\.1/g,'Пять отношений').replace(/архиве v1\.1/g,'сравнении пяти отношений').replace(/\s+/g,' ').trim();const available=new Map();for(const paragraph of document.querySelectorAll('p')){const key=text(paragraph);available.set(key,(available.get(key)||0)+1)}return [...source.querySelectorAll('p')].map(text).filter(value=>{const count=available.get(value)||0;if(count)available.set(value,count-1);return !count})},data.contest.v12.ui_fragment);
      assert.deepEqual(paragraphRetention,[],'All saved scientific explanations and qualifications remain available');
      await checkLayout(page,'expanded evidence');
      await activate(page.locator('.evidence > .section-head [data-focus-search]'));await page.waitForFunction(()=>document.activeElement.id==='search');
      assert.deepEqual(await retainedState(page),preserved,'Evidence navigation preserves the complete working state');await checkTerritory(page,1);record.actions.push('all evidence, old comparisons and same territory on return');
      for(const view of ['map','analogs','dynamics']){await activate(page.locator(`#view-${view}`));await page.locator('.atlas-workspace').scrollIntoViewIfNeeded();const name=`${view}-${viewport.width}.png`;await page.screenshot({path:path.join(output,name),animations:'disabled'});record.screenshots.push(name)}
      const motion=await page.evaluate(()=>{const select=document.getElementById('quick');document.body.dispatchEvent(new PointerEvent('pointerdown',{bubbles:true}));for(const value of ['400','401']){select.value=value;select.dispatchEvent(new Event('change',{bubbles:true}))}return {selected:select.value,text:document.getElementById('identity').textContent,rows:[...document.querySelectorAll('#neighbors tr')].map(n=>n.dataset.entityId),active:document.getElementById('territory-inspector').getAnimations().length,events:window.__MOTION_REVIEW__.slice(-20)}});
      assert.equal(motion.selected,'401');assert(motion.text.includes(data.entities[401].name));assert.deepEqual(motion.rows,expectedNeighbors(401,1));assert(motion.active>0);
      assert(motion.events.some(e=>e.kind==='cancel'&&e.id==='territory-inspector'));assert(motion.events.some(e=>e.kind==='start'&&e.id==='territory-inspector'&&e.options.duration===200));record.motion=motion;record.actions.push('immediate facts and cancellation during rapid territory changes');
      await page.evaluate(()=>{for(const id of ['view-map','view-dynamics','view-analogs'])document.getElementById(id).click()});await page.waitForTimeout(250);
      assert.equal(await page.locator('#view-analogs').getAttribute('aria-selected'),'true');assert(await page.locator('#workspace-analogs').isVisible());assert.equal(await page.locator('[data-leaving]').count(),0);
      await page.locator('#view-analogs').focus();await page.locator('#view-analogs').press('ArrowRight');assert.equal(await page.locator('#view-dynamics').getAttribute('aria-selected'),'true');assert.equal(await page.evaluate(()=>document.activeElement.id),'view-dynamics');
      assert.equal(await page.locator('#workspace-dynamics').evaluate(n=>n.getAnimations().length),0,'Keyboard navigation is immediate');record.actions.push('interrupted view transitions and keyboard tabs');
      await page.emulateMedia({reducedMotion:'reduce'});await page.locator('#quick').selectOption('250');await activate(page.locator('#view-map'));
      assert.equal(await page.locator('#territory-inspector').evaluate(n=>n.getAnimations().length),0);assert.equal(await page.locator('.view-indicator').evaluate(n=>getComputedStyle(n).transitionDuration),'0s');
      const selected=page.locator('#territory-map path.map-shape.selected');await selected.focus();await selected.press('ArrowRight');
      assert.equal(await page.locator('#territory-map .map-outline-focus').evaluate(n=>getComputedStyle(n).display),'inline');
      record.actions.push('static reduced-motion and map keyboard focus');
      await checkMapWorkspace(page,data,record);record.actions.push('preserved map geometry, complete outlines, pointer, zoom, pan, keyboard and historical monthly network');
      const example=page.locator('button[data-select-entity]').first(),exampleId=await example.getAttribute('data-select-entity');await activate(example);
      assert.equal(data.entities[+await page.locator('#quick').inputValue()].id,exampleId);assert.equal(await page.locator('#exclude').inputValue(),'-1');await checkTerritory(page);record.actions.push('scientific examples select the named territory and its exact main-model neighbors');
      const deepLink=new URL(base.href);deepLink.hash='network-title';await page.goto(deepLink.href,{waitUntil:'load'});await page.waitForFunction(()=>window.__ATLAS_READY__===true);assert(await page.locator('#network-title').isVisible());
      assert.equal(await page.locator('#proof-model').evaluate(n=>n.open),true);assert.equal(await page.locator('#archive-v11').evaluate(n=>n.open),true);record.actions.push('direct deep evidence anchor');
      assert.deepEqual(record.errors,[]);assert.deepEqual(record.httpErrors,[]);assert.deepEqual(record.failedRequests,[]);assert.deepEqual(record.blockedRequests,[]);record.status='PASS';save();
    } catch(error) {record.status='FAIL';if(record.userPath?.status==='RUNNING')record.userPath.status='INCOMPLETE';record.failure=String(error.stack||error);const name=`failure-${viewport.width}.png`;await page.screenshot({path:path.join(output,name),fullPage:true,animations:'disabled'}).then(()=>record.screenshots.push(name)).catch(error=>record.screenshotError=String(error));save();throw error}
    finally {const name=`trace-${viewport.width}.zip`;await context.tracing.stop({path:path.join(output,name)}).then(()=>record.trace=name).catch(error=>record.traceError=String(error));save();await context.close()}
  }
}
(async()=>{let browser;try{browser=await chromium.launch({headless:true});await runViewports(browser);report.status='PASS';save();console.log(JSON.stringify(report))}catch(error){report.status='FAIL';report.failure=String(error.stack||error);save();throw error}finally{if(browser)await browser.close()}})().catch(error=>{console.error(error);process.exitCode=1});
