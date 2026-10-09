const assert=require('node:assert/strict');
const path=require('node:path');
const fs=require('node:fs');
const zlib=require('node:zlib');

async function checkMobileStory(page,data,output){
 const id=await page.evaluate(()=>window.__ATLAS_STORY__.selectedId),entity=data.entities.find(e=>e.id===id);
 assert(entity,'Selected mobile identity is canonical');
 const format=new Intl.NumberFormat('ru-RU',{maximumFractionDigits:2});
 const totals=entity.totals.slice(0,12).filter(Number.isFinite).sort((a,b)=>a-b);
 const values=[(totals[Math.floor((totals.length-1)/2)]+totals[Math.ceil((totals.length-1)/2)])/2,...entity.annual_ratios];
 const rows=page.locator('.story-mobile-profile-row');assert.equal(await rows.count(),6);
 for(let j=0;j<6;j++){
  const row=rows.nth(j);assert.equal(await row.locator('dt').textContent(),j===0?'Общий показатель':data.categories[j-1]);
  assert.equal(await row.locator('dd strong').textContent(),format.format(values[j]));
  assert.equal(await row.locator('.story-mobile-unit').textContent(),j===0?'руб./жителя в месяц':'% от общего показателя');
 }
 assert.equal(await page.locator('[data-frame-step="1"] > canvas').isVisible(),false,'Profile uses readable HTML rather than a scaled desktop chart');
 const counts=[0,1,2,3].map(g=>data.entities.filter(e=>e.v12_type===g).length);
 assert.deepEqual(await page.locator('[data-frame-step="3"] .story-mobile-legend span').allTextContents(),counts.map((n,g)=>`Группа ${g+1} · ${format.format(n)} МО`));
 assert.equal(await page.locator('[data-frame-step="4"] .story-mobile-legend span').count(),4);
 assert.deepEqual(await page.locator('.story-mobile-time-ticks span').allTextContents(),[data.periods[0],data.periods[12],data.periods.at(-1)].map(p=>p.slice(0,7)));
 assert((await page.locator('[data-frame-step="5"] .story-mobile-time-axis').textContent()).includes('руб./жителя в месяц'));
 const sizes=await page.locator('.story-mobile-labels dt,.story-mobile-labels dd strong,.story-mobile-unit,.story-mobile-range span,.story-mobile-legend span,.story-mobile-time-axis,.story-mobile-time-ticks span,.story-mobile-caption,.story-mobile-identity,.story-mobile-frame > .story-chart-note').evaluateAll(nodes=>nodes.map(n=>({text:n.textContent,fontSize:parseFloat(getComputedStyle(n).fontSize)})));
 assert(sizes.every(n=>n.fontSize>=12),'Every essential mobile label is at least12CSSpx');
 assert.equal(await page.evaluate(()=>document.documentElement.scrollWidth>innerWidth+1),false);
 const frames=[];
 for(let s=0;s<6;s++){
  const frame=page.locator(`[data-frame-step="${s}"]`);
  assert((await frame.locator('.story-mobile-identity').textContent()).includes(entity.name));
  await frame.scrollIntoViewIfNeeded();
  const filename=`mobile-scene-${s}.png`;await frame.screenshot({path:path.join(output,filename)});frames.push(filename);
 }
 return {status:'PASS',selectedId:id,profileValues:values,minimumFontPx:Math.min(...sizes.map(n=>n.fontSize)),groupCounts:counts,frames};
}
module.exports={checkMobileStory};
if(require.main===module)(async()=>{
 const {chromium}=require(process.env.PLAYWRIGHT_PATH||'playwright');
 const htmlFile=path.resolve(process.argv[2]||'docs/index.html'),output=path.resolve(process.argv[3]||'artifacts/mobile-readability');fs.mkdirSync(output,{recursive:true});
 const html=fs.readFileSync(htmlFile,'utf8'),asset=html.match(/<script src="(assets\/payload-[a-f0-9]{64}\.js)"><\/script>/);
 const source=asset?fs.readFileSync(path.join(path.dirname(htmlFile),asset[1]),'utf8'):html;
 const packed=source.match(/\{"encoding":"gzip-base64","data":"([A-Za-z0-9+/=]+)"\}/);assert(packed);
 const data=JSON.parse(zlib.gunzipSync(Buffer.from(packed[1],'base64')));
 const browser=await chromium.launch({headless:true,...(process.env.CHROME_PATH?{executablePath:process.env.CHROME_PATH}:{})});
 const errors=[],result={status:'RUNNING'};
 try{
  const page=await browser.newPage({viewport:{width:390,height:844},reducedMotion:'reduce'});page.on('pageerror',e=>errors.push(e.message));
  const began=Date.now();await page.goto(process.env.TEST_BASE_URL);await page.waitForFunction(()=>window.__ATLAS_READY__&&window.__ATLAS_STORY__);result.readyMs=Date.now()-began;
  Object.assign(result,await checkMobileStory(page,data,output));
  await page.locator('#territory-options > summary').click();await page.locator('#quick').selectOption('0');
  assert.equal(await page.evaluate(()=>window.__ATLAS_STORY__.selectedId),data.entities[0].id);
  assert((await page.locator('[data-frame-step="1"] .story-mobile-identity').textContent()).includes(data.entities[0].name));result.afterSelectionId=data.entities[0].id;
  assert.equal(await page.evaluate(()=>window.__ATLAS_STORY__.animating),false);assert.equal(await page.locator('[data-story-step]').count(),6);
  await page.emulateMedia({media:'print'});await page.evaluate(()=>window.dispatchEvent(new Event('beforeprint')));assert.equal(await page.locator('.story-mobile-profile-row').count(),6);
  assert.deepEqual(errors,[]);result.errors=errors;
 }catch(error){result.status='FAIL';result.failure=error.stack;throw error;}
 finally{fs.writeFileSync(path.join(output,'result.json'),JSON.stringify(result,null,2));await browser.close();}
 console.log(JSON.stringify(result));
})().catch(error=>{console.error(error);process.exitCode=1});
