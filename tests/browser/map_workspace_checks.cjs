const assert=require('node:assert/strict');
module.exports=async function checkMapWorkspace(page,data,record){
const checks=[];record.mapChecks=checks;const add=(name,pass,details)=>{checks.push({name,pass:Boolean(pass),details});assert(pass,name)};
const desktop=page.viewportSize().width>600;
const activate=locator=>desktop?locator.click():locator.tap();
async function selectQuick(value){const summary=page.locator('#territory-options > summary');await activate(summary);await page.selectOption('#quick',value);await activate(summary);await page.locator('#territory-map').scrollIntoViewIfNeeded()}
const view=page=>page.locator('#territory-map').evaluate(e=>({box:[e.viewBox.baseVal.x,e.viewBox.baseVal.y,e.viewBox.baseVal.width,e.viewBox.baseVal.height],scale:e.getScreenCTM().a}));
const selectedIndex=page=>page.locator('#territory-map .map-shape.selected').getAttribute('data-index');
const outline=async(page,kind)=>page.locator('#territory-map').evaluate((root,kind)=>{
  const group=root.querySelector(`.map-outline-${kind}`),paths=group?[...group.children]:[];
  return {visible:!!group&&getComputedStyle(group).display!=='none',d:paths.map(p=>p.getAttribute('d')),decorative:!!group&&group.getAttribute('aria-hidden')==='true'&&paths.length===2&&paths.every(p=>p.getAttribute('fill')==='none'&&getComputedStyle(p).fill==='none'&&getComputedStyle(p).pointerEvents==='none'&&p.getAttribute('aria-hidden')==='true'&&!p.hasAttribute('data-index')&&!p.hasAttribute('tabindex')&&!p.hasAttribute('role')),painted:paths.some(p=>getComputedStyle(p).stroke!=='none'&&parseFloat(getComputedStyle(p).strokeWidth)>0),scaleSafe:paths.every(p=>getComputedStyle(p).vectorEffect==='non-scaling-stroke')};
},kind);
async function checkSelected(page,name){
  const shape=page.locator('#territory-map .map-shape.selected'),d=await shape.getAttribute('d'),current=await outline(page,'selected');
  add(name,current.visible&&current.decorative&&current.painted&&current.scaleSafe&&current.d.every(v=>v===d),{index:await selectedIndex(page)});
}
async function visibleTarget(page,excluded){
  return page.locator('#territory-map').evaluate((svg,excluded)=>{
    const box=svg.getBoundingClientRect();
    for(const p of svg.querySelectorAll('path.map-shape[data-index]')){
      if(p.dataset.index===''||p.dataset.index===excluded)continue;
      const b=p.getBBox(),m=p.getScreenCTM();if(b.width*m.a<12||b.height*m.d<10||b.width*m.a>400)continue;
      for(const fx of [.5,.35,.65,.2,.8])for(const fy of [.5,.35,.65,.2,.8]){
        const q=new DOMPoint(b.x+b.width*fx,b.y+b.height*fy).matrixTransform(m);
        if(q.x<box.left+12||q.x>box.right-20||q.y<Math.max(box.top+20,70)||q.y>Math.min(box.bottom-12,innerHeight-12))continue;
        if(document.elementFromPoint(q.x,q.y)===p)return{x:q.x,y:q.y,index:p.dataset.index,d:p.getAttribute('d')};
      }
    }
    throw new Error('No visible municipal interior');
  },excluded);
}
async function target(page){return page.evaluate(()=>{
  const svg=document.querySelector('#territory-map'), box=svg.getBoundingClientRect();
  const selected=document.querySelector('#territory-map .selected')?.dataset.index;
  const paths=[...svg.querySelectorAll('.map-shape')].filter(p=>p.dataset.index!==selected);
  for(const p of paths){
    const b=p.getBBox(),m=p.getScreenCTM();
    if(b.width*m.a<12||b.height*m.d<10||b.width*m.a>400)continue;
    for(const fx of [.5,.35,.65,.2,.8])for(const fy of [.5,.35,.65,.2,.8]){
      const q=new DOMPoint(b.x+b.width*fx,b.y+b.height*fy).matrixTransform(m);
      if(q.x<box.left+12||q.x>box.right-80||q.y<Math.max(box.top+20,70)||q.y>Math.min(box.bottom-12,innerHeight-12))continue;
      if(document.elementFromPoint(q.x,q.y)===p)return {x:q.x,y:q.y,index:p.dataset.index};
    }
  }
  throw new Error('No visible distinct municipal interior for real click');
});}

await activate(page.locator('#view-map'));await page.locator('#territory-map').scrollIntoViewIfNeeded();
await page.locator('.map-options').evaluate(d=>d.open=true);
await page.evaluate(()=>{window.__firstMapPath=document.querySelector('#territory-map .map-shape')});
    const sourcePaths=data.contest.map.paths,entities=data.entities;
    const actual=await page.locator('#territory-map > .map-shape').evaluateAll(ps=>ps.map(p=>({d:p.getAttribute('d'),index:p.dataset.index})));
    const byId=new Map(entities.map((e,i)=>[e.id,String(i)]));
    add('all_source_geometries_and_order_exact',actual.length===sourcePaths.length&&actual.every((p,i)=>p.d===sourcePaths[i].d&&p.index===(byId.get(sourcePaths[i].id)??'')),{paths:actual.length});
    add('holes_use_evenodd_fill_rule',await page.locator('#territory-map > .map-shape').evaluateAll(ps=>ps.every(p=>p.getAttribute('fill-rule')==='evenodd'&&getComputedStyle(p).fillRule==='evenodd')));
    add('no_native_map_titles',await page.locator('#territory-map title').count()===0);
    await page.evaluate(()=>{window.__mapOriginalPaths=[...document.querySelectorAll('#territory-map > .map-shape')]});
    add('outlines_last_above_every_base_path',await page.locator('#territory-map').evaluate(svg=>svg.lastElementChild?.classList.contains('map-outline-layer')&&[...svg.querySelectorAll('.map-shape')].every(p=>p.parentElement===svg&&(p.compareDocumentPosition(svg.lastElementChild)&Node.DOCUMENT_POSITION_FOLLOWING))));
    for(const kind of ['selected','hover','focus'])add(kind+'_outlines_are_decorative',(await outline(page,kind)).decorative);
    await checkSelected(page,'initial_selection_full_outline');

if(desktop){
  for(const zoom of [1,2,4,8]){
    await page.click('#map-zoom-reset');
    for(let z=1;z<zoom;z*=2)await page.click('#map-zoom-in');
    const point=await target(page),before=await view(page);
    await page.mouse.move(point.x,point.y);
    add('hover_disabled_by_default_zoom_'+zoom,await page.locator('#map-tooltip').isHidden());
    await page.mouse.click(point.x,point.y);
    const after=await view(page);
    add('real_pointer_click_zoom_'+zoom,await page.locator('#territory-map .selected').getAttribute('data-index')===point.index,{expected:point.index});
    add('selection_preserves_view_zoom_'+zoom,before.box.every((v,i)=>Math.abs(v-after.box[i])<1e-7),{before:before.box,after:after.box});
  }
  add('map_geometry_nodes_reused_on_selection',await page.evaluate(()=>window.__firstMapPath===document.querySelector('#territory-map .map-shape')));
    // Reproduce stale DOM focus: pointerdown is cancelled while zoomed.
    const first=await visibleTarget(page,await selectedIndex(page));
    await page.mouse.click(first.x,first.y);
    await page.locator(`#territory-map .map-shape[data-index="${first.index}"]`).focus();
    await page.mouse.wheel(0,-550);
    await page.waitForTimeout(200);
    const second=await visibleTarget(page,first.index);
    await page.mouse.click(second.x,second.y);
    add('sequential_pointer_selection',await selectedIndex(page)===second.index&&first.index!==second.index);
    await checkSelected(page,'second_pointer_selection_full_outline');
    const stale=await page.evaluate(index=>{const p=document.querySelector(`#territory-map .map-shape[data-index="${index}"]`),s=getComputedStyle(p);return{active:document.activeElement===p,selected:p.classList.contains('selected'),outline:s.outlineStyle,width:s.outlineWidth}},first.index);
    add('old_focused_path_has_no_UA_rectangle',stale.active&&!stale.selected&&(stale.outline==='none'||parseFloat(stale.width)===0),stale);
    add('pointer_selection_hides_keyboard_focus',!(await outline(page,'focus')).visible);
    const hovered=await visibleTarget(page,await selectedIndex(page));
    await page.mouse.move(hovered.x,hovered.y);
    const hover=await outline(page,'hover');
    add('hover_full_contour_and_pointer_hits_base',hover.visible&&hover.d.every(d=>d===hovered.d)&&await page.evaluate(({x,y,index})=>document.elementFromPoint(x,y)?.dataset.index===index,hovered));
    await page.mouse.move(1,1);
    add('pointer_leave_removes_hover_contour',!(await outline(page,'hover')).visible);
  const beforeSearch=await view(page);
  const current=await page.locator('#quick').inputValue();
  await selectQuick(current==='0'?'1':'0');
  const afterSearch=await view(page);
  add('dropdown_selection_preserves_zoom',beforeSearch.box.every((v,i)=>Math.abs(v-afterSearch.box[i])<1e-7),{before:beforeSearch.box,after:afterSearch.box});
  await page.click('#map-zoom-reset');await page.click('#map-zoom-in');await page.click('#map-zoom-in');
  const before=await view(page),rect=await page.locator('#territory-map').boundingBox();
  const sx=rect.x+rect.width*.55,sy=rect.y+rect.height*.55,dx=72,dy=28;
  const selected=await page.locator('#territory-map .selected').getAttribute('data-index');
  await page.mouse.move(sx,sy);await page.mouse.down();await page.mouse.move(sx+dx,sy+dy,{steps:9});await page.mouse.up();
  const after=await view(page);
  const errorX=(after.box[0]-before.box[0])*before.scale+dx,errorY=(after.box[1]-before.box[1])*before.scale+dy;
  add('pan_tracks_cursor_within_one_pixel',Math.abs(errorX)<1&&Math.abs(errorY)<1,{errorX,errorY});
  add('drag_does_not_select',await page.locator('#territory-map .selected').getAttribute('data-index')===selected);
  add('no_text_selection_after_drag',await page.evaluate(()=>!getSelection().toString()));
  const anchored=await view(page);
  await page.selectOption('#map-year','2024');await page.selectOption('#map-mode','relative');
  const redrawn=await view(page);
  add('year_and_mode_preserve_pan',anchored.box.every((v,i)=>Math.abs(v-redrawn.box[i])<1e-7));
  await page.locator('#map-hover-tips').check();const tooltipPoint=await target(page);await page.mouse.move(tooltipPoint.x,tooltipPoint.y);
  add('opt_in_hover_shows_description',await page.locator('#map-tooltip').isVisible());await page.locator('#map-hover-tips').uncheck();
  add('opt_out_hides_tooltip',await page.locator('#map-tooltip').evaluate(e=>e.hidden&&getComputedStyle(e).pointerEvents==='none'));
  const cancelRect=await page.locator('#territory-map').boundingBox();
  const cx=cancelRect.x+cancelRect.width*.5,cy=cancelRect.y+cancelRect.height*.5;
  await page.evaluate(()=>document.querySelector('#territory-map').addEventListener('pointerdown',e=>window.__mapQAId=e.pointerId,{once:true}));
  await page.mouse.move(cx,cy);await page.mouse.down();
  const primaryBefore=await view(page);
  await page.evaluate(({x,y})=>document.querySelector('#territory-map').dispatchEvent(new PointerEvent('pointermove',{pointerId:999,isPrimary:false,clientX:x+60,clientY:y+30,bubbles:true})),{x:cx,y:cy});
  const secondaryAfter=await view(page);
  add('secondary_pointer_does_not_move_drag',primaryBefore.box.every((v,i)=>Math.abs(v-secondaryAfter.box[i])<1e-7));
  await page.mouse.move(cx+20,cy+10,{steps:3});
  await page.evaluate(()=>document.querySelector('#territory-map').dispatchEvent(new PointerEvent('pointercancel',{pointerId:window.__mapQAId,isPrimary:true,bubbles:true})));
  const canceled=await view(page);
  await page.mouse.move(cx+70,cy+30,{steps:3});await page.mouse.up();
  const afterCanceled=await view(page);
  add('pointercancel_releases_drag',canceled.box.every((v,i)=>Math.abs(v-afterCanceled.box[i])<1e-7));

}else{
await page.locator('#map-zoom-reset').tap();await page.locator('#map-zoom-in').tap();await page.locator('#map-zoom-in').tap();
await page.locator('#map-hover-tips').tap();
await page.locator('#territory-map').scrollIntoViewIfNeeded();
const point=await target(page),before=await view(page);await page.touchscreen.tap(point.x,point.y);const after=await view(page);
add('mobile_real_touch_selects_zoomed_municipality',await selectedIndex(page)===point.index);
add('mobile_touch_preserves_zoom',before.box.every((v,i)=>Math.abs(v-after.box[i])<1e-7));
add('mobile_touch_leaves_no_sticky_hover',!(await outline(page,'hover')).visible&&await page.locator('#map-tooltip').isHidden());
await page.locator('#map-hover-tips').tap();
}
    await page.locator('#territory-map .map-shape.selected').focus();
    await page.keyboard.press('ArrowRight');
    let keyboard=await page.evaluate(()=>({index:document.activeElement.dataset.index,d:document.activeElement.getAttribute('d'),outline:getComputedStyle(document.activeElement).outlineStyle}));
    const focus=await outline(page,'focus');
    add('arrow_focus_visible_by_full_contour',focus.visible&&focus.d.every(d=>d===keyboard.d)&&keyboard.outline==='none'&&keyboard.index!==await selectedIndex(page));
    const everyFocus=await page.locator('#territory-map').evaluate((root,index)=>{
      const group=root.querySelector('.map-outline-focus'),paths=[...root.querySelectorAll(':scope > .map-shape[data-index]:not([data-index=""])')];
      let failure=null;
      for(const p of paths){p.focus({preventScroll:true});if(getComputedStyle(group).display==='none'||[...group.children].some(o=>o.getAttribute('d')!==p.getAttribute('d'))){failure=p.dataset.index;break}}
      root.querySelector(`.map-shape[data-index="${index}"]`).focus({preventScroll:true});
      return {count:paths.length,failure};
    },keyboard.index);
    add('full_contour_for_every_focusable_territory',everyFocus.count===sourcePaths.filter(p=>byId.has(p.id)).length&&everyFocus.failure===null,everyFocus);
    await page.keyboard.press('Enter');
    add('Enter_selects_focused_territory',await selectedIndex(page)===keyboard.index);
    await checkSelected(page,'keyboard_selection_full_outline');
    await page.keyboard.press('End');await page.keyboard.press(' ');
    const expectedLast=sourcePaths.filter(p=>byId.has(p.id)).at(-1);
    add('End_and_Space_select_last',await selectedIndex(page)===byId.get(expectedLast.id));
    await page.keyboard.press('Home');await page.keyboard.press('Enter');
    add('Home_and_Enter_select_first',await selectedIndex(page)===byId.get(sourcePaths.find(p=>byId.has(p.id)).id));
    const islands=sourcePaths.filter(p=>(p.d.match(/M/g)||[]).length>1).slice(0,3);
    for(const item of islands){await selectQuick(byId.get(item.id));await checkSelected(page,'multipart_selection_'+item.id)}
    await activate(page.locator('#map-zoom-in'));
    await checkSelected(page,'zoom_preserves_complete_outline');
    await page.selectOption('#map-year','2024');await page.selectOption('#map-mode','relative');
    await page.locator('#map-transitions').check();
    await checkSelected(page,'year_mode_and_transitions_preserve_outline');
    add('outlines_stay_opaque_when_base_is_dimmed',await page.locator('#territory-map .map-outline-selected path').evaluateAll(ps=>ps.every(p=>getComputedStyle(p).opacity==='1')));
    add('base_nodes_geometry_and_order_unchanged_after_updates',await page.locator('#territory-map').evaluate((root,expected)=>{const ps=[...root.querySelectorAll(':scope > .map-shape')];return ps.length===expected.length&&ps.every((p,i)=>p===window.__mapOriginalPaths[i]&&p.getAttribute('d')===expected[i].d&&p.dataset.index===expected[i].index)},actual));
    await activate(page.locator('#map-zoom-reset'));await page.locator('#territory-map').scrollIntoViewIfNeeded();
    const fit=await view(page),expectedFit=data.contest.map.viewBox||[0,0,1000,420];
    add('fit_restores_original_viewbox',fit.box.every((v,i)=>Math.abs(v-expectedFit[i])<1e-7),{expected:expectedFit,actual:fit.box});
    // Real Tab entry uses the selected territory as the only map tab stop.
    await page.locator('#map-hover-tips').focus();
    let entered=false;
    for(let i=0;i<12;i++){await page.keyboard.press('Tab');entered=await page.evaluate(()=>document.activeElement.matches('#territory-map .map-shape'));if(entered)break}
    add('Tab_enters_map_with_contour_focus',entered&&(await outline(page,'focus')).visible);
    await page.keyboard.press('Tab');
    add('Tab_leaving_map_removes_focus_contour',!(await outline(page,'focus')).visible);

await page.locator('#proof-model').evaluate(d=>d.open=true);await page.locator('#archive-v11').evaluate(d=>d.open=true);
const historicalModels={frozen:{labels:Object.fromEntries(data.entities.map(e=>[e.id,data.validation?e.reference_cluster:data.reference?e.annual_cluster:e.pilot_cluster]))},...data.contest.map_models};
const expectedGroups=Object.values(historicalModels).map(model=>{const labels=Object.values(model.labels||{}).filter(label=>label!==null),counts=new Map();for(const label of labels)counts.set(label,(counts.get(label)||0)+1);return{coverage:labels.length,sizes:[...counts.entries()].sort((a,b)=>a[0]-b[0]).map(([label,count])=>`${label+1}: ${count}`).join(' · ')}});
const displayedGroups=await page.locator('#comparison-models tbody tr').evaluateAll(rows=>rows.map(row=>({coverage:Number(row.cells[2].textContent),sizes:row.cells[3].textContent})));
add('all_saved_model_coverages_and_group_sizes_exact',JSON.stringify(displayedGroups)===JSON.stringify(expectedGroups),{models:displayedGroups.length});
    await page.selectOption('#ego-period','1');
    add('monthly_network_has_15_edges',await page.locator('#ego-network line[data-edge-id]').count()===15);
    add('monthly_network_has_16_nodes',await page.locator('#ego-network g[data-node-index]').count()===16);
    if(data?.contest?.comparability){
      const rows=await page.locator('#neighbors tr').evaluateAll(nodes=>nodes.map(n=>({id:n.dataset.entityId,ratio:+n.dataset.expenseRatio,type:n.querySelector('.territory-type')?.textContent})));
      const selected=+await page.locator('#quick').inputValue(),own=data.contest.comparability[data.entities[selected].id].expense_2023;
      add('neighbor_level_ratios_match_source',rows.length===15&&rows.every(r=>Math.abs(r.ratio-data.contest.comparability[r.id].expense_2023/own)<1e-12));
      add('neighbor_types_match_source',rows.every(r=>r.type===data.contest.comparability[r.id].municipal_district_type));
      const nodes=await page.locator('#ego-network g[data-node-index]').evaluateAll(ns=>ns.map(n=>+n.dataset.nodeIndex)),peers=nodes.filter(i=>i!==selected),period=1,scale=data.meta.ratio_iqr;
      const top=i=>new Set(data.entities.map((e,j)=>({j,d:i===j?Infinity:e.ratios[period].reduce((sum,v,c)=>sum+((Math.log(v)-Math.log(data.entities[i].ratios[period][c]))/scale[c])**2/5,0)})).sort((a,b)=>a.d-b.d||data.entities[a.j].id.localeCompare(data.entities[b.j].id)).slice(0,15).map(v=>v.j));
      const topSets=new Map(peers.map(i=>[i,top(i)])),expected=[];
      const edgeKey=(a,b)=>[a,b].sort().join('|');
      for(let a=0;a<peers.length;a++)for(let b=a+1;b<peers.length;b++)if(topSets.get(peers[a]).has(peers[b])||topSets.get(peers[b]).has(peers[a]))expected.push(edgeKey(data.entities[peers[a]].id,data.entities[peers[b]].id));
      const actual=await page.locator('#ego-network line[data-context-edge]').evaluateAll(es=>es.map(e=>[e.dataset.contextSource,e.dataset.contextTarget].sort().join('|')));
      add('induced_context_edges_match_full_cohort_knn',JSON.stringify(actual.sort())===JSON.stringify(expected.sort()),{edges:actual.length});
      add('context_edges_do_not_capture_pointer',await page.locator('#ego-network line[data-context-edge]').evaluateAll(es=>es.every(e=>getComputedStyle(e).pointerEvents==='none')));
    }else{add('comparability_payload_available',false);}

    add('monthly_network_reports_retained_links',(await page.locator('#ego-summary').textContent()).includes('Сохранилось'));
    const neighbor=page.locator('#ego-network g[data-node-index]').nth(1),neighborId=await neighbor.getAttribute('data-node-index');
    await activate(neighbor);
    add('network_node_selects_municipality',await page.locator('#quick').inputValue()===neighborId);

if(data.contest.round2){add('temporal_grid_displays_all_five_weights',await page.locator('#temporal-summary table').first().locator('tbody tr').count()===5);add('temporal_grid_has_three_seeds_each',data.contest.temporal.variants.length===5&&data.contest.temporal.variants.every(v=>v.seeds===3));add('regional_comparison_displays_three_models',await page.locator('#temporal-summary table').nth(1).locator('tbody tr').count()===3)}
add('main_joint_model_survives_historical_evidence',await page.locator('#map-model').inputValue()==='v12_types'&&await page.locator('#method').inputValue()==='v12');
record.mapChecks=checks;return checks;
};
