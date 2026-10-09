// Local authorized preview; does not replace frozen acceptance or remote CI.
const assert=require('node:assert/strict');
const fs=require('node:fs');
const path=require('node:path');
const zlib=require('node:zlib');
const {checkMobileStory}=require('./mobile_story_checks.cjs');
const {chromium}=require(process.env.PLAYWRIGHT_PATH||'playwright');
const base=process.env.TEST_BASE_URL;
assert(base && /^https?:/.test(base),'Supply HTTP preview URL');
async function runStoryChecks(sharedBrowser, targetOutput){
 const output=path.resolve(targetOutput||process.argv[2]||'artifacts/atlas-preview/browser');fs.mkdirSync(output,{recursive:true});
 const browser=sharedBrowser||await chromium.launch({headless:true,...(process.env.CHROME_PATH?{executablePath:process.env.CHROME_PATH}:{})});
 const report={scope:'Local Chromium emulation; science unchanged',checks:[],frames:[],errors:[]};
 const page=await browser.newPage({viewport:{width:1440,height:900}});
 await page.addInitScript(()=>{const request=window.requestAnimationFrame.bind(window),cancel=window.cancelAnimationFrame.bind(window),pending=new Set();window.__STORY_RAF_REQUESTS__=0;window.__STORY_LONG_TASKS__=[];window.requestAnimationFrame=callback=>{window.__STORY_RAF_REQUESTS__++;let id;id=request(time=>{pending.delete(id);callback(time);});pending.add(id);return id;};window.cancelAnimationFrame=id=>{pending.delete(id);cancel(id);};window.__STORY_PENDING_RAF__=()=>pending.size;try{new PerformanceObserver(list=>window.__STORY_LONG_TASKS__.push(...list.getEntries().map(e=>({start:e.startTime,duration:e.duration})))).observe({type:'longtask',buffered:true});}catch{}});
 page.on('pageerror',e=>report.errors.push(e.message));page.on('console',e=>{if(e.type()==='error')report.errors.push(e.text());});
 const requests=[];page.on('request',r=>requests.push(r.url()));
 try{
  const started=Date.now();await page.goto(base);await page.waitForFunction(()=>window.__ATLAS_READY__&&window.__ATLAS_STORY__);report.readyMs=Date.now()-started;
  report.timing=await page.evaluate(()=>({boot:window.__ATLAS_BOOT_TIMING__,story:window.__ATLAS_STORY_METRICS__,navigation:performance.getEntriesByType('navigation').map(e=>({responseEnd:e.responseEnd,domContentLoaded:e.domContentLoadedEventEnd,load:e.loadEventEnd})),resources:performance.getEntriesByType('resource').map(e=>({name:e.name,duration:e.duration,bytes:e.transferSize})),longTasks:window.__STORY_LONG_TASKS__}));
  const htmlFile=path.resolve(process.env.TEST_HTML_FILE||'artifacts/atlas-preview/index.html'),html=fs.readFileSync(htmlFile,'utf8'),payload=html.match(/<script src="(assets\/payload-[a-f0-9]{64}\.js)"><\/script>/);report.loadedBytes=fs.statSync(htmlFile).size+(payload?fs.statSync(path.join(path.dirname(htmlFile),payload[1])).size:0);assert(report.loadedBytes<=8*1024*1024,'HTML+payload <=8MiB');
  assert.equal(await page.locator('[data-story-step]').count(),6);assert.equal(await page.locator('#territory-map .map-shape').count(),2016);
  const identity=await page.evaluate(()=>window.__ATLAS_STORY__.selectedId);
  await page.screenshot({path:path.join(output,'desktop-hero.png')});
  console.log(JSON.stringify({firstFrame:path.join(output,'desktop-hero.png'),readyMs:report.readyMs,bytes:report.loadedBytes,timing:report.timing.story}));
  for(let s=0;s<6;s++){
   await page.evaluate(s=>window.__ATLAS_STORY__.activate(s),s);
   await page.waitForTimeout(600);
   assert.equal(await page.evaluate(()=>window.__ATLAS_STORY__.selectedId),identity);
   assert.equal(await page.evaluate(()=>window.__ATLAS_STORY__.animating),false);
   // Observer can change scene while offscreen; scroll to its semantic final frame.
   await page.locator(`[data-story-step="${s}"]`).scrollIntoViewIfNeeded();
   await page.evaluate(s=>window.__ATLAS_STORY__.activate(s),s);await page.waitForTimeout(600);
   await page.screenshot({path:path.join(output,`desktop-scene-${s}.png`)});report.frames.push(`desktop-scene-${s}.png`);
  }
  const interaction=await page.evaluate(()=>{const at=performance.now();window.__ATLAS_STORY__.activate(2);return performance.now()-at;});report.interactionSyncMs=interaction;assert(interaction<=200,'Interaction feedback <=200ms');await page.waitForTimeout(90);await page.evaluate(()=>window.__ATLAS_STORY__.activate(4));await page.waitForTimeout(100);await page.screenshot({path:path.join(output,'reversal-during.png')});await page.waitForTimeout(550);assert.equal(await page.evaluate(()=>window.__ATLAS_STORY__.animating),false);
  const idleBefore=await page.evaluate(()=>window.__STORY_RAF_REQUESTS__);await page.waitForTimeout(350);assert.equal(await page.evaluate(()=>window.__STORY_RAF_REQUESTS__),idleBefore,'No idle RAF');assert.equal(await page.evaluate(()=>window.__STORY_PENDING_RAF__()),0);
  await page.locator('#story-next').focus();await page.keyboard.press('Enter');assert.equal(await page.evaluate(()=>window.__ATLAS_STORY__.animating),false,'Keyboard settles immediately');
  const fonts=await page.evaluate(()=>[...document.querySelectorAll('h2,h3')].map(n=>{const s=getComputedStyle(n);return {family:s.fontFamily,weight:s.fontWeight,synthesis:s.fontSynthesis};}));assert(fonts.every(s=>s.family.includes('Cinzel RU')&&+s.weight>=600&&s.synthesis.includes('weight')));report.fonts=fonts[0];
  const overflow=()=>page.evaluate(()=>document.documentElement.scrollWidth>innerWidth+1);
  assert.equal(await overflow(),false);
  await page.emulateMedia({reducedMotion:'reduce'});await page.evaluate(()=>window.__ATLAS_STORY__.activate(1));assert.equal(await page.evaluate(()=>window.__ATLAS_STORY__.animating),false);
  await page.emulateMedia({reducedMotion:'no-preference'});
  await page.locator('#territory-options > summary').click();await page.locator('#quick').selectOption('0');assert.equal(await page.evaluate(()=>window.__ATLAS_STORY__.selectedId),'tid_196');
  await page.locator('#view-analogs').click();assert.equal(await page.locator('#neighbors tr').count(),15);
  await page.setViewportSize({width:390,height:844});await page.evaluate(()=>scrollTo(0,0));await page.waitForTimeout(200);
  assert.equal(await overflow(),false);assert.equal(await page.locator('.story-mobile-frame canvas').count(),6);
  await page.screenshot({path:path.join(output,'mobile-hero.png')});
  await page.locator('#story-profile').scrollIntoViewIfNeeded();await page.screenshot({path:path.join(output,'mobile-profile.png')});
  const savedSource=payload?fs.readFileSync(path.join(path.dirname(htmlFile),payload[1]),'utf8'):html;
  const savedPacked=savedSource.match(/\{"encoding":"gzip-base64","data":"([A-Za-z0-9+/=]+)"\}/);assert(savedPacked);
  report.mobileReadability=await checkMobileStory(page,JSON.parse(zlib.gunzipSync(Buffer.from(savedPacked[1],'base64'))),output);
  const small=await page.evaluate(()=>[...document.querySelectorAll('#atlas-story button,#atlas-story a')].filter(n=>{const r=n.getBoundingClientRect();return r.width&&r.height&&r.height<43.5;}).map(n=>n.textContent));assert.deepEqual(small,[]);
  const context=await browser.newContext({viewport:{width:1440,height:900},reducedMotion:'reduce'});const offline=await context.newPage();await offline.route('**/*',r=>new URL(r.request().url()).origin===new URL(base).origin?r.continue():r.abort());await offline.goto(base+'#story-types');await offline.waitForFunction(()=>window.__ATLAS_READY__);assert.equal(await offline.evaluate(()=>window.__ATLAS_STORY__.step),3);await offline.emulateMedia({media:'print'});await offline.evaluate(()=>window.dispatchEvent(new Event('beforeprint')));assert.equal(await offline.locator('.story-mobile-frame canvas').count(),6);await context.close();
  assert.deepEqual(report.errors,[]);assert(requests.every(url=>new URL(url).origin===new URL(base).origin||url.startsWith('data:')));assert(report.readyMs<=5000,'Desktop ready <=5s');report.requiredRequests=requests;report.checks.push('six scenes and original 2016 shapes','identity retained','interruptible and settled without RAF','Cinzel RU synthetic700','1440/390 no page overflow','15 analogs and selection','reduced motion and keyboard immediate','direct anchor and remote-blocked startup','print six static frames','44px story controls','8MiB budget and ready <=5s');report.status='PASS';
 }catch(e){report.status='FAIL';report.failure=e.stack;throw e;}finally{fs.writeFileSync(path.join(output,'result.json'),JSON.stringify(report,null,2));await page.close();if(!sharedBrowser)await browser.close();}
 return report;
}
module.exports={runStoryChecks};
if(require.main===module)runStoryChecks().catch(e=>{console.error(e);process.exitCode=1;});
