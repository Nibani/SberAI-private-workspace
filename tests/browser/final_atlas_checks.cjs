// Both original acceptance suites, one Chromium process, sequential contexts.
const fs=require('node:fs');
const path=require('node:path');
const {chromium}=require(process.env.PLAYWRIGHT_PATH||'playwright');
process.env.TEST_HTML_FILE=path.resolve(process.argv[2]||'docs/index.html');
const {runMainChecks}=require('./main_model_regressions.cjs');
const {runStoryChecks}=require('./story_checks.cjs');
const output=path.resolve(process.argv[3]||'artifacts/atlas-final/browser/main');
(async()=>{
 const browser=await chromium.launch({headless:true,...(process.env.CHROME_PATH?{executablePath:process.env.CHROME_PATH}:{})});
 const final={status:'RUNNING',html:process.env.TEST_HTML_FILE,scope:'One local Chromium process; original story and main functional suites, no skipped assertions'};
 try{
  const story=await runStoryChecks(browser,path.join(path.dirname(output),'story'));final.story={status:story.status,readyMs:story.readyMs,bytes:story.loadedBytes};
  const main=await runMainChecks(browser);final.main={status:main.status,viewports:main.checks.map(c=>({viewport:c.viewport,status:c.status,errors:c.errors}))};
  final.status='PASS';console.log(JSON.stringify(final));
 }catch(error){final.status='FAIL';final.failure=String(error.stack||error);throw error;}
 finally{fs.mkdirSync(path.dirname(output),{recursive:true});fs.writeFileSync(path.join(path.dirname(output),'integrated-result.json'),JSON.stringify(final,null,2));await browser.close();}
})().catch(error=>{console.error(error);process.exitCode=1;});
