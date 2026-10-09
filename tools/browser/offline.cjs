// Cold file:// atlas under Chromium offline mode: no server or third-party runtime.
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const {pathToFileURL} = require('node:url');
const {chromium} = require(process.env.PLAYWRIGHT_PATH || 'playwright');
const file = path.resolve(process.argv[2] || 'docs/index.html');
const output = path.resolve(process.argv[3] || 'artifacts/browser-offline');
fs.mkdirSync(output, {recursive:true});
const report = {status:'RUNNING', scope:'Cold file://, offline network, desktop/mobile Chromium emulation', checks:[]};
const save = () => fs.writeFileSync(path.join(output,'result.json'),JSON.stringify(report,null,2)+'\n');
(async()=>{
  const browser = await chromium.launch({headless:true,...(process.env.CHROME_PATH?{executablePath:process.env.CHROME_PATH}:{})});
  try {
    for (const viewport of [{width:1440,height:900},{width:390,height:844}]) {
      const context = await browser.newContext({viewport,offline:true,serviceWorkers:'block',reducedMotion:'reduce'});
      const page = await context.newPage();
      await page.addInitScript(()=>{
        window.__OFFLINE_INPUT_TRACE__=[];
        for(const type of ['pointerdown','pointerup','click','blur','change','keydown'])document.addEventListener(type,event=>{
          window.__OFFLINE_INPUT_TRACE__.push({type,target:event.target.id||event.target.tagName,key:event.key||null,x:event.clientX??null,y:event.clientY??null,focus:document.activeElement?.id||null,scrollY,time:performance.now()});
          if(window.__OFFLINE_INPUT_TRACE__.length>40)window.__OFFLINE_INPUT_TRACE__.shift();
        },true);
      });
      const errors=[], remote=[];
      report.activeViewport=viewport;report.errors=errors;report.remote=remote;
      page.on('pageerror',error=>errors.push(String(error)));
      page.on('console',message=>{if(message.type()==='error')errors.push(message.text())});
      page.on('request',request=>{if(/^https?:/.test(request.url()))remote.push(request.url())});
      try {
      const start=Date.now();
      await page.goto(pathToFileURL(file).href,{waitUntil:'load'});
      try {
        await page.waitForFunction(()=>window.__ATLAS_READY__===true,null,{timeout:30000});
      } catch(error) {
        report.timeoutContext={viewport,errors,remote,
          status:await page.locator('#status').textContent({timeout:2000}).catch(()=>null),
          ready:await page.evaluate(()=>window.__ATLAS_READY__===true).catch(()=>null)};
        save();throw error;
      }
      const ready_ms=Date.now()-start;
      assert.equal(await page.locator('#territory-map path.map-shape').count(),2016);
      // Enter the workspace through its visible user route, then let hash focus settle.
      await page.locator('.hero-links [data-focus-search]').click();
      await page.waitForFunction(()=>location.hash==='#territory-title'&&document.activeElement?.id==='search');
      const settlePresentation=()=>page.evaluate(()=>new Promise(resolve=>requestAnimationFrame(()=>requestAnimationFrame(resolve))));
      await settlePresentation();
      const selectedIndex=250;
      const searchLabel=await page.locator('#territories option').nth(selectedIndex).getAttribute('value');
      assert(searchLabel,'Offline search exposes a complete territory label');
      await page.locator('#search').fill(searchLabel);
      await page.locator('#search').press('Enter');
      assert.equal(await page.locator('#quick').inputValue(),String(selectedIndex),'Offline search selects the requested territory');
      await page.locator('#view-analogs').click();
      assert.equal(await page.locator('#view-analogs').getAttribute('aria-selected'),'true');
      assert.equal(await page.locator('#neighbors tr').count(),15);
      await page.locator('#exclude').selectOption('0');
      assert.equal(await page.locator('#neighbors tr').count(),15);
      await page.locator('#view-dynamics').click();
      const record={viewport,ready_ms,errors,remote};
      assert.deepEqual(errors,[]);assert.deepEqual(remote,[]);
      await page.screenshot({path:path.join(output,`offline-${viewport.width}.png`),fullPage:true,animations:'disabled'});
      report.checks.push(record);save();
      } catch(error) {
        report.failureContext=await page.evaluate(()=>({
          ready:window.__ATLAS_READY__===true,hash:location.hash,focus:document.activeElement?.id||document.activeElement?.tagName,
          selectedIndex:document.getElementById('quick')?.value,scroll:{x:scrollX,y:scrollY},
          tabs:[...document.querySelectorAll('[data-view]')].map(tab=>({id:tab.id,selected:tab.getAttribute('aria-selected'),rect:tab.getBoundingClientRect().toJSON()})),
          panels:[...document.querySelectorAll('[data-workspace-panel]')].map(panel=>({id:panel.id,hidden:panel.hidden,inert:panel.inert})),
          inputTrace:window.__OFFLINE_INPUT_TRACE__||[]
        })).catch(diagnosticError=>({error:String(diagnosticError)}));
        const screenshot=`offline-failure-${viewport.width}.png`;
        await page.screenshot({path:path.join(output,screenshot),animations:'disabled'}).then(()=>report.failureScreenshot=screenshot).catch(screenshotError=>report.screenshotError=String(screenshotError));
        save();throw error;
      } finally {await context.close();}
    }
    report.status='PASS';save();
  } finally {await browser.close()}
})().catch(error=>{report.status='FAIL';report.failure=String(error.stack||error);save();console.error(error);process.exitCode=1});
