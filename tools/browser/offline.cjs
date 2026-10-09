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
      const errors=[], remote=[];
      report.activeViewport=viewport;report.errors=errors;report.remote=remote;
      page.on('pageerror',error=>errors.push(String(error)));
      page.on('console',message=>{if(message.type()==='error')errors.push(message.text())});
      page.on('request',request=>{if(/^https?:/.test(request.url()))remote.push(request.url())});
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
      await page.locator('#search').fill('Москва');
      assert((await page.locator('#quick option').count())>0,'Offline search works');
      await page.locator('#view-analogs').click();
      assert.equal(await page.locator('#neighbors tr').count(),15);
      await page.locator('#exclude').selectOption('0');
      assert.equal(await page.locator('#neighbors tr').count(),15);
      await page.locator('#view-dynamics').click();
      const record={viewport,ready_ms,errors,remote};
      assert.deepEqual(errors,[]);assert.deepEqual(remote,[]);
      await page.screenshot({path:path.join(output,`offline-${viewport.width}.png`),fullPage:true,animations:'disabled'});
      report.checks.push(record);save();await context.close();
    }
    report.status='PASS';save();
  } finally {await browser.close()}
})().catch(error=>{report.status='FAIL';report.failure=String(error.stack||error);save();console.error(error);process.exitCode=1});
