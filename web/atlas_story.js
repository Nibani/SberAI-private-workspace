/* Native story renderer. Coordinates are presentation; saved measurements never change. */
(() => {
  'use strict';
  const palette = ['#14705b', '#49779b', '#ae7b37', '#916183'];
  const ink = '#1e382f', muted = '#60726b', paper = '#faf8f1';
  const titles = ['География · 2023', 'Шесть измерений · медианы 2023', '15 аналогов · признаки 2023', 'Расходные группы · 2023', 'Группы на карте · 2023', 'Наблюдения · 2023–2024'];
  const notes = [
    'Точки соответствуют территориям исходной карты. Охват: 2 016 МО, сопоставленных между годами.',
    'Шесть отдельных шкал: общий показатель в рублях и отношения категорий в %. Высота распределения — число МО; отметка — выбранное МО.',
    '15 ближайших соседей по шести стандартизованным признакам. Условная круговая раскладка, расстояния здесь не географические.',
    'Строка — группа основной модели. Горизонталь — относительный уровень расходов (логарифм, шкала IQR 2023). Вертикальная раскладка внутри строки условная.',
    'Геометрия прежняя, цвет — группа модели 2023 года. Назначение не устанавливает специализацию и не измеряет поток.',
    'Общий показатель, руб./жителя в месяц. Светлые линии — выбранные аналоги; тёмная — выбранная территория. Только наблюдённые месяцы.'
  ];
  const median = values => { const v = values.filter(Number.isFinite).sort((a,b)=>a-b); return v.length ? (v[Math.floor((v.length-1)/2)]+v[Math.ceil((v.length-1)/2)])/2 : null; };
  const fmt = value => Number.isFinite(value) ? new Intl.NumberFormat('ru-RU',{maximumFractionDigits:2}).format(value) : 'нет данных';
  const clamp = (v,a,b) => Math.max(a,Math.min(b,v));
  const $ = id => document.getElementById(id);
  const node = (tag,text,className) => { const e=document.createElement(tag); if(text!==undefined)e.textContent=text; if(className)e.className=className; return e; };
  window.SberAtlasStory = {init};
  function init(D, api) {
    const initializedAt=performance.now();
    const root=$('atlas-story'), canvas=$('story-canvas');
    if(!root || !canvas || !D.entities?.length)return;
    const entities=D.entities, byId=new Map(entities.map((e,i)=>[e.id,i]));
    const paths=D.contest?.map?.paths || [], viewBox=D.contest?.map?.viewBox || [0,0,1000,420];
    const chapters=[...root.querySelectorAll('[data-story-step]')];
    const anchors=new Map(), shapes=[], mapLayers=new Map();
    // Saved municipal geometry consists of absolute M/L/Z commands. Its vertex
    // bounds give the same presentation anchor as SVG getBBox, without duplicating
    // 2 016 complex SVG elements or forcing another large layout.
    for(const p of paths){
      const coordinates=p.d.match(/[-+]?(?:\d*\.)?\d+(?:[eE][-+]?\d+)?/g)?.map(Number)||[];
      let left=Infinity,right=-Infinity,top=Infinity,bottom=-Infinity;
      for(let k=0;k<coordinates.length;k+=2){left=Math.min(left,coordinates[k]);right=Math.max(right,coordinates[k]);top=Math.min(top,coordinates[k+1]);bottom=Math.max(bottom,coordinates[k+1]);}
      anchors.set(p.id,{x:(left+right)/2,y:(top+bottom)/2});shapes.push({id:p.id,path:new Path2D(p.d)});
    }
    const featureNames=['Общий показатель',...D.categories];
    const measurements=entities.map(e=>[median(e.totals.slice(0,12)),...(e.annual_ratios || D.categories.map((_,j)=>median(e.ratios.slice(0,12).map(v=>v[j]))))]);
    const ranges=featureNames.map((_,j)=>{const values=measurements.map(v=>v[j]).filter(Number.isFinite);return {min:Math.min(...values),max:Math.max(...values)};});
    const histograms=ranges.map((r,j)=>{const bins=Array(48).fill(0);for(const values of measurements){if(Number.isFinite(values[j]))bins[Math.min(47,Math.floor(48*(values[j]-r.min)/(r.max-r.min||1)))]++;}return bins;});
    let index=api.index(), step=0, current=[], target=[], raf=0, visible=true, folio=null;
    let userReduced=false;try{userReduced=localStorage.getItem('sber-atlas-less-motion')==='1';}catch{}
    const media=matchMedia('(prefers-reduced-motion: reduce)'), mobile=matchMedia('(max-width: 760px)');
    const reduced=()=>userReduced||media.matches||mobile.matches;
    let graphList=[], graphSet=new Set(), graphRank=new Map();
    const graph=()=>{graphList=(entities[index].v12_neighbors || []).map(id=>byId.get(id)).filter(i=>i!==undefined);graphSet=new Set([index,...graphList]);graphRank=new Map(graphList.map((i,k)=>[i,k]));};
    graph();
    // A PCA projection is supplied by data preparation, never fitted in the UI.
    const layout=D.contest?.story_layout;
    const layoutCoordinates=layout?.coordinates,layoutIds=layout?.entity_ids;
    const globalLayout=Array.isArray(layoutCoordinates)&&Array.isArray(layoutIds)
      &&layoutCoordinates.length===entities.length&&layoutIds.length===entities.length
      &&new Set(layoutIds).size===entities.length&&layoutIds.every(id=>byId.has(id))
      &&layoutCoordinates.every(p=>Array.isArray(p)&&p.length===2&&p.every(Number.isFinite))
      &&typeof layout.method==='string'&&layout.method.length>0;
    const layoutMap=new Map(globalLayout?layoutIds.map((id,i)=>[id,layoutCoordinates[i]]):[]);
    const edges=[],edgeKeys=new Set();
    if(globalLayout)entities.forEach((e,i)=>(e.v12_neighbors||[]).forEach(id=>{const j=byId.get(id);if(j===undefined||j===i)return;const key=i<j?`${i}:${j}`:`${j}:${i}`;if(!edgeKeys.has(key)){edgeKeys.add(key);edges.push([i,j]);}}));
    const layoutBounds=globalLayout?{xmin:Math.min(...layoutCoordinates.map(p=>p[0])),xmax:Math.max(...layoutCoordinates.map(p=>p[0])),ymin:Math.min(...layoutCoordinates.map(p=>p[1])),ymax:Math.max(...layoutCoordinates.map(p=>p[1]))}:null;
    const group=i=>entities[i].v12_type ?? entities[i].reference_cluster ?? 0;
    const geo=i=>{const a=anchors.get(entities[i].id);return a?{x:46+908*(a.x-viewBox[0])/viewBox[2],y:110+382*(a.y-viewBox[1])/viewBox[3]}:{x:500,y:300};};
    function positions(s){
      const counters=Array(4).fill(0), levels=entities.map(e=>e.v12_features?.[5]).filter(Number.isFinite);
      const lo=Math.min(...levels),hi=Math.max(...levels);
      const temporalValues=[...graphList,index].flatMap(i=>entities[i].totals).filter(Number.isFinite),temporalLow=Math.min(...temporalValues),temporalHigh=Math.max(...temporalValues);
      return entities.map((e,i)=>{
        const g=group(i), a=geo(i); let x=a.x,y=a.y,opacity=1,r=1.7;
        if(s===1){const j=i%6,v=measurements[i][j],range=ranges[j];x=220+710*(v-range.min)/(range.max-range.min||1);y=80+j*76-((Math.floor(i/6)%7)-3)*1.6;opacity=.22;r=1.4;}
        if(s===2){if(globalLayout){const p=layoutMap.get(e.id);if(p){x=80+840*(p[0]-layoutBounds.xmin)/(layoutBounds.xmax-layoutBounds.xmin||1);y=95+410*(p[1]-layoutBounds.ymin)/(layoutBounds.ymax-layoutBounds.ymin||1);}r=graphSet.has(i)?5.5:2;opacity=graphSet.has(i)?1:.68;if(i===index)r=9;}else if(i===index){x=500;y=287;r=10;}else if(graphSet.has(i)){const k=graphRank.get(i),angle=-Math.PI/2+k*2*Math.PI/graphList.length;x=500+300*Math.cos(angle);y=287+185*Math.sin(angle);r=6;}else{opacity=.035;r=1;}}
        if(s===3){x=190+730*((e.v12_features?.[5]??0)-lo)/(hi-lo||1);y=108+g*108+((counters[g]++%17)-8)*2.5;r=2.4;opacity=.6;}
        if(s===5){x=930;y=495-395*(e.totals.at(-1)-temporalLow)/(temporalHigh-temporalLow||1);opacity=graphSet.has(i)?1:0;r=graphSet.has(i)?3:0;}
        if(i===index && s!==2){opacity=1;r=6;}
        return {x,y,opacity,r,color:(s===0||s===1||s===5)?ink:palette[g%4],id:e.id};
      });
    }
    const scaleCanvas=c=>{const rect=c.getBoundingClientRect(),dpr=Math.min(devicePixelRatio||1,2);if(!rect.width)return null;const width=Math.round(rect.width*dpr),height=Math.round(rect.width*.6*dpr);if(c.width!==width||c.height!==height){c.width=width;c.height=height;}const ctx=c.getContext('2d');ctx.setTransform(c.width/1000,0,0,c.height/600,0,0);return ctx;};
    function map(ctx,s){
      const key=`${s}:${ctx.canvas.width}:${ctx.canvas.height}`;
      let layer=mapLayers.get(key);
      if(!layer){
        layer=document.createElement('canvas');layer.width=ctx.canvas.width;layer.height=ctx.canvas.height;
        const painted=layer.getContext('2d');painted.setTransform(layer.width/1000,0,0,layer.height/600,0,0);
        painted.translate(46,110);painted.scale(908/viewBox[2],382/viewBox[3]);painted.translate(-viewBox[0],-viewBox[1]);painted.lineWidth=.35;
        for(const shape of shapes){const i=byId.get(shape.id);painted.fillStyle=s===4&&i!==undefined?palette[group(i)%4]:'#d9e2db';painted.globalAlpha=s===4?.38:.7;painted.fill(shape.path);painted.strokeStyle=paper;painted.stroke(shape.path);}
        if(mapLayers.size>=4)mapLayers.delete(mapLayers.keys().next().value);mapLayers.set(key,layer);
      }
      ctx.drawImage(layer,0,0,1000,600);
      // Selection changes only its exact outline; the complex map remains cached.
      const shape=shapes.find(p=>p.id===entities[index].id);
      if(shape){ctx.save();ctx.translate(46,110);ctx.scale(908/viewBox[2],382/viewBox[3]);ctx.translate(-viewBox[0],-viewBox[1]);ctx.strokeStyle=ink;ctx.lineWidth=2.3;ctx.stroke(shape.path);ctx.restore();}
    }
    function text(ctx,value,x,y,opts={}){if(ctx.canvas.closest('.story-mobile-frame'))return;ctx.fillStyle=opts.color||muted;ctx.font=`${opts.bold?'600':'400'} ${opts.size||14}px "Segoe UI",sans-serif`;ctx.textAlign=opts.align||'left';ctx.fillText(value,x,y);}
    function backdrop(ctx,s){
      if(s===0||s===4){map(ctx,s);text(ctx,'Геометрия карты · сопоставленные муниципалитеты',48,550);}
      if(s===1){histograms.forEach((bins,j)=>{const y=80+j*76,max=Math.max(...bins);ctx.beginPath();ctx.moveTo(220,y);bins.forEach((n,k)=>ctx.lineTo(220+710*k/47,y-43*n/max));ctx.lineTo(930,y);ctx.closePath();ctx.fillStyle=['#d4e5dd','#dbe5ed','#e5dfcb','#e8dce4','#dce6d4','#e9dfd2'][j];ctx.fill();ctx.strokeStyle='#c3cec6';ctx.beginPath();ctx.moveTo(220,y);ctx.lineTo(930,y);ctx.stroke();text(ctx,featureNames[j],30,y-15,{size:14,bold:true});text(ctx,j===0?'руб./жителя в месяц':'% от общего показателя',30,y+5,{size:12});const v=measurements[index][j],range=ranges[j],x=220+710*(v-range.min)/(range.max-range.min||1);ctx.strokeStyle=ink;ctx.lineWidth=2;ctx.beginPath();ctx.moveTo(x,y-48);ctx.lineTo(x,y+4);ctx.stroke();text(ctx,fmt(v),x,y-53,{align:x>820?'right':'left',bold:true,color:ink});text(ctx,fmt(range.min),220,y+18,{size:11});text(ctx,fmt(range.max),930,y+18,{size:11,align:'right'});});}
      if(s===2){text(ctx,'Связи по 6 признакам · цвет расходной группы',500,38,{align:'center',bold:true});text(ctx,globalLayout?`PCA: условная 2D-проекция · ${edges.length} реальных связей`:'Центр: выбранная территория · 15 соседей',500,565,{align:'center'});}
      if(s===3){for(let g=0;g<4;g++){const y=108+g*108;ctx.fillStyle=['#edf3ed','#ecf1f5','#f5efe1','#f3ecf1'][g];ctx.fillRect(170,y-40,774,80);const n=entities.filter((_,i)=>group(i)===g).length;text(ctx,`Группа ${g+1}`,30,y-7,{bold:true,color:palette[g]});text(ctx,`${n} МО`,30,y+15);}text(ctx,'Ниже ← относительный уровень расходов → выше',190,555);}
      if(s===4){for(let g=0;g<4;g++){ctx.fillStyle=palette[g];ctx.fillRect(48+g*223,558,9,9);text(ctx,`Группа ${g+1}`,65+g*223,567);}}
      if(s===5){const values=[...graphList,index].flatMap(i=>entities[i].totals).filter(Number.isFinite),lo=Math.min(...values),hi=Math.max(...values);for(let k=0;k<4;k++){const y=100+k*395/3;ctx.strokeStyle='#dce1d9';ctx.lineWidth=1;ctx.beginPath();ctx.moveTo(100,y);ctx.lineTo(930,y);ctx.stroke();text(ctx,fmt(hi-k*(hi-lo)/3),88,y+4,{align:'right',size:12});}text(ctx,'руб./жителя',28,55,{bold:true});for(const i of [...graphList,index]){const vals=entities[i].totals;ctx.strokeStyle=i===index?ink:'#c3d2c8';ctx.lineWidth=i===index?3:1.1;ctx.beginPath();let drawing=false;vals.forEach((v,k)=>{if(!Number.isFinite(v)){drawing=false;return;}const x=100+830*k/(D.periods.length-1),y=495-395*(v-lo)/(hi-lo||1);drawing?ctx.lineTo(x,y):ctx.moveTo(x,y);drawing=true;});ctx.stroke();if(i===index){vals.forEach((v,k)=>{if(!Number.isFinite(v))return;ctx.beginPath();ctx.arc(100+830*k/(D.periods.length-1),495-395*(v-lo)/(hi-lo||1),3,0,2*Math.PI);ctx.fillStyle=ink;ctx.fill();});}}for(const k of [0,6,12,18,23])text(ctx,D.periods[k]?.slice(0,7)||'',100+830*k/(D.periods.length-1),530,{align:'center',size:12});}
    }
    function draw(c,s,points){const ctx=scaleCanvas(c);if(!ctx)return;ctx.clearRect(0,0,1000,600);if(c.id!=='hero-canvas'){ctx.fillStyle=paper;ctx.fillRect(0,0,1000,600);}backdrop(ctx,s);if(s===2){if(globalLayout){ctx.lineWidth=.45;ctx.globalAlpha=.09;for(const [i,j] of edges){ctx.strokeStyle=palette[group(i)%4];ctx.beginPath();ctx.moveTo(points[i].x,points[i].y);ctx.lineTo(points[j].x,points[j].y);ctx.stroke();}ctx.globalAlpha=1;}ctx.strokeStyle='#4c8573';ctx.lineWidth=1.6;for(const i of graphList){ctx.beginPath();ctx.moveTo(points[index].x,points[index].y);ctx.lineTo(points[i].x,points[i].y);ctx.stroke();}}
      points.forEach((p,i)=>{if(p.opacity<.01||p.r===0)return;ctx.globalAlpha=p.opacity;ctx.fillStyle=p.color;ctx.beginPath();ctx.arc(p.x,p.y,p.r,0,Math.PI*2);ctx.fill();if(i===index){ctx.globalAlpha=1;ctx.strokeStyle='#b28a42';ctx.lineWidth=2;ctx.beginPath();ctx.arc(p.x,p.y,p.r+5,0,Math.PI*2);ctx.stroke();}});ctx.globalAlpha=1;
      if(s===2){if(!globalLayout)graphList.forEach((i,k)=>{const p=points[i];text(ctx,String(k+1).padStart(2,'0'),p.x,p.y-12,{align:'center',bold:true,color:ink});});text(ctx,'Выбрано',points[index].x,points[index].y+31,{align:'center',bold:true,color:ink});}
    }
    function numeric(s,container){container.replaceChildren();const e=entities[index];container.append(node('p',`${e.name} · ${e.region}`));if(s===1){const dl=node('dl',undefined,'story-value-list');measurements[index].forEach((v,j)=>{const row=node('div');row.append(node('dt',featureNames[j]),node('dd',`${fmt(v)} ${j===0?'руб./жителя в месяц':'%'}`));dl.append(row);});container.append(dl);}else if(s===2){const ol=node('ol');graphList.forEach(i=>{const item=node('li'),button=node('button',`${entities[i].name} · ${entities[i].region}`);button.type='button';button.addEventListener('click',()=>api.select(i));item.append(button);ol.append(item);});container.append(ol);}else if(s===5){const table=node('table'),thead=node('thead'),head=node('tr');head.append(node('th','Месяц'),node('th','руб./жителя'));thead.append(head);const tbody=node('tbody');D.periods.forEach((p,k)=>{const tr=node('tr');tr.append(node('td',p.slice(0,7)),node('td',fmt(e.totals[k])));tbody.append(tr);});table.append(thead,tbody);container.append(table);}else{container.append(node('p',`Назначенная группа основной модели 2023: ${group(index)+1}. Относительный уровень (логарифм, IQR): ${fmt(e.v12_features?.[5])}.`));}container.append(node('p',notes[s]));}
    function mobileLabels(host,s){
      const e=entities[index],labels=node('div',undefined,'story-mobile-labels');
      labels.append(node('p',`Выбрано: ${e.name} · ${e.region}`,'story-mobile-identity'));
      if(s===1){
        const list=node('dl',undefined,'story-mobile-profile');
        measurements[index].forEach((v,j)=>{
          const row=node('div',undefined,'story-mobile-profile-row'),value=node('dd'),unit=j===0?'руб./жителя в месяц':'% от общего показателя';
          row.dataset.featureIndex=String(j);row.append(node('dt',featureNames[j]));
          value.append(node('strong',fmt(v)),node('span',unit,'story-mobile-unit'));row.append(value);
          const distribution=node('div',undefined,'story-mobile-distribution'),bins=histograms[j],peak=Math.max(...bins);
          distribution.setAttribute('aria-hidden','true');
          bins.forEach(n=>{const bar=node('i');bar.style.height=`${100*n/(peak||1)}%`;distribution.append(bar);});
          if(Number.isFinite(v)){const marker=node('b');marker.style.left=`${100*clamp((v-ranges[j].min)/(ranges[j].max-ranges[j].min||1),0,1)}%`;distribution.append(marker);}
          const limits=node('div',undefined,'story-mobile-range');limits.append(node('span',fmt(ranges[j].min)),node('span',fmt(ranges[j].max)));row.append(distribution,limits);list.append(row);
        });
        labels.append(list,node('p','Распределения по 2 016 МО · отметка — выбранная территория · медианы 2023 года','story-mobile-caption'));
      }else{
        labels.append(node('p',titles[s],'story-mobile-caption'));
        if(s===2||s===3||s===4){
          const legend=node('div',undefined,'story-mobile-legend');
          for(let g=0;g<4;g++){const label=node('span',`Группа ${g+1}${s===3?` · ${fmt(entities.filter((_,i)=>group(i)===g).length)} МО`:''}`);label.style.setProperty('--story-group-color',palette[g]);legend.append(label);}labels.append(legend);
          if(s===3)labels.append(node('p','Ниже ← относительный уровень расходов → выше','story-mobile-caption'));
        }
        if(s===5){
          const values=[...graphList,index].flatMap(i=>entities[i].totals).filter(Number.isFinite),axis=node('div',undefined,'story-mobile-time-axis');
          axis.append(node('span',`${fmt(Math.min(...values))}–${fmt(Math.max(...values))} руб./жителя в месяц`));labels.append(axis);
          const ticks=node('div',undefined,'story-mobile-time-ticks');for(const k of [0,12,D.periods.length-1])ticks.append(node('span',D.periods[k].slice(0,7)));labels.append(ticks);
          labels.append(node('p',`${D.periods.at(-1).slice(0,7)} · выбранная территория: ${fmt(e.totals.at(-1))} руб./жителя в месяц`,'story-mobile-caption'));
        }
      }
      host.append(labels);
    }
    function mobileFrames(){if(!mobile.matches && !matchMedia('print').matches)return;cancelMobile();chapters.forEach((article,s)=>{const host=article.querySelector('.story-mobile-frame');host.dataset.frameStep=String(s);host.replaceChildren();const c=node('canvas');c.setAttribute('aria-hidden','true');host.append(c);if(s!==1)draw(c,s,positions(s));mobileLabels(host,s);host.append(node('p',notes[s],'story-chart-note'));const details=node('details'),summary=node('summary','Числа и подписи'),body=node('div');details.append(summary,body);host.append(details);numeric(s,body);});}
    function sync(){root.dataset.storyActive=String(step);$('story-view-label').textContent=titles[step];$('story-chart-note').textContent=notes[step];$('story-entity-name').textContent=entities[index].name;$('story-entity-region').textContent=entities[index].region;$('story-progress').textContent=`${String(step+1).padStart(2,'0')} / 06`;$('story-prev').disabled=step===0;$('story-next').disabled=step===5;root.querySelectorAll('[data-story-target]').forEach(a=>{a.setAttribute('aria-current',+a.dataset.storyTarget===step?'step':'false');});chapters.forEach((a,s)=>a.classList.toggle('is-current',s===step));numeric(step,$('story-numeric'));$('story-motion').setAttribute('aria-pressed',String(userReduced));$('story-motion').textContent=userReduced?'Движение выключено':'Меньше движения';}
    function copySheet(){const sheet=document.createElement('canvas');sheet.width=canvas.width;sheet.height=canvas.height;sheet.getContext('2d').drawImage(canvas,0,0);return sheet;}
    function paintFolio(turn,t){
      const ctx=scaleCanvas(canvas);if(!ctx)return;
      ctx.clearRect(0,0,1000,600);ctx.drawImage(turn.after,0,0,1000,600);
      const backwards=turn.backwards,edge=backwards?1000*t:1000*(1-t),lift=Math.sin(Math.PI*t),skew=24*lift;
      const top=edge-skew,bottom=edge+skew,side=backwards?1000:0,sign=backwards?-1:1,curl=54*lift;
      // A previous physical sheet is peeled away. Neither plot's entities move.
      ctx.save();ctx.beginPath();ctx.moveTo(side,0);ctx.lineTo(top,0);ctx.lineTo(bottom,600);ctx.lineTo(side,600);ctx.closePath();ctx.clip();ctx.drawImage(turn.before,0,0,1000,600);ctx.restore();
      if(curl>0.1){
        const shadow=ctx.createLinearGradient(edge,0,edge+sign*(curl+28),0);shadow.addColorStop(0,'rgba(30,56,47,.18)');shadow.addColorStop(1,'rgba(30,56,47,0)');ctx.fillStyle=shadow;
        ctx.beginPath();ctx.moveTo(top,0);ctx.lineTo(top+sign*(curl+28),0);ctx.lineTo(bottom+sign*(curl+28),600);ctx.lineTo(bottom,600);ctx.closePath();ctx.fill();
        const light=ctx.createLinearGradient(edge,0,edge+sign*curl,0);light.addColorStop(0,'#d8d7ca');light.addColorStop(.25,'#fffef6');light.addColorStop(.75,paper);light.addColorStop(1,'#e6e3d6');ctx.fillStyle=light;
        ctx.beginPath();ctx.moveTo(top,0);ctx.lineTo(top+sign*curl*.72,12*lift);ctx.lineTo(bottom+sign*curl,600-12*lift);ctx.lineTo(bottom,600);ctx.closePath();ctx.fill();
        ctx.strokeStyle='rgba(96,114,107,.2)';ctx.lineWidth=.8;ctx.beginPath();ctx.moveTo(top,0);ctx.lineTo(bottom,600);ctx.stroke();
      }
    }
    function settle(){cancelAnimationFrame(raf);raf=0;folio=null;current=target.map(p=>({...p}));draw(canvas,step,current);root.dataset.storySettled='true';}
    function activate(s,animate=true){
      const began=performance.now(),previous=step,next=clamp(s,0,5),hasSheet=current.length>0;
      cancelAnimationFrame(raf);raf=0;
      const before=hasSheet&&next!==previous&&!reduced()&&visible&&animate?copySheet():null;
      step=next;sync();target=positions(step);current=target.map(p=>({...p}));
      if(!before){settle();return;}
      // Only the two complete drawings are composited; numbers and coordinates are final immediately.
      draw(canvas,step,current);const turn={before,after:copySheet(),backwards:next<previous,start:began};folio=turn;
      paintFolio(turn,0);root.dataset.storySettled='false';
      function frame(time){if(folio!==turn)return;const u=clamp((time-turn.start)/520,0,1),t=u*u*(3-2*u);paintFolio(turn,t);if(u<1)raf=requestAnimationFrame(frame);else{raf=0;folio=null;const ctx=scaleCanvas(canvas);if(ctx){ctx.clearRect(0,0,1000,600);ctx.drawImage(turn.after,0,0,1000,600);}root.dataset.storySettled='true';}}
      raf=requestAnimationFrame(frame);
    }
    function go(s,animate=true){activate(s,animate);chapters[step].scrollIntoView({block:'center',behavior:'auto'});}
    root.querySelectorAll('[data-story-target]').forEach(a=>a.addEventListener('click',event=>activate(+a.dataset.storyTarget,event.detail!==0)));
    $('story-prev').addEventListener('click',event=>go(step-1,event.detail!==0));$('story-next').addEventListener('click',event=>go(step+1,event.detail!==0));
    $('story-motion').addEventListener('click',()=>{userReduced=!userReduced;try{localStorage.setItem('sber-atlas-less-motion',userReduced?'1':'0');}catch{}activate(step,false);});
    media.addEventListener('change',()=>activate(step,false));mobile.addEventListener('change',()=>{mobileFrames();activate(step,false);});
    function hero(){const c=$('hero-canvas');if(c){draw(c,0,positions(0));$('hero-selected-label').textContent=entities[index].name;}}
    window.addEventListener('atlas-selection',e=>{index=e.detail.index;graph();activate(step,false);mobileFrames();hero();});
    const chapterObserver=new IntersectionObserver(entries=>{if(mobile.matches)return;const candidates=entries.filter(e=>e.isIntersecting);if(candidates.length){const distance=entry=>Math.abs(entry.boundingClientRect.top+entry.boundingClientRect.height/2-innerHeight*.4);const best=candidates.sort((a,b)=>distance(a)-distance(b))[0];const s=+best.target.dataset.storyStep;if(s!==step)activate(s);}}, {rootMargin:'-25% 0px -35% 0px',threshold:0});chapters.forEach(c=>chapterObserver.observe(c));
    const seenMobile=new WeakSet(),mobileArrivals=new Set();
    const mobileObserver=new IntersectionObserver(entries=>{if(!mobile.matches)return;for(const entry of entries){if(!entry.isIntersecting){for(const arrival of mobileArrivals)if(arrival.article===entry.target){arrival.motion.cancel();arrival.cover.remove();mobileArrivals.delete(arrival);}continue;}if(seenMobile.has(entry.target))continue;seenMobile.add(entry.target);if(userReduced||media.matches||document.body.dataset.interaction==='keyboard')continue;const c=entry.target.querySelector('.story-mobile-frame canvas');if(c?.animate&&c.getBoundingClientRect().height){const cover=node('div',undefined,'story-mobile-leaf');cover.setAttribute('aria-hidden','true');cover.style.height=`${c.getBoundingClientRect().height}px`;c.parentElement.append(cover);const motion=cover.animate([{transform:'perspective(700px) rotateY(0deg)'},{transform:'perspective(700px) rotateY(-42deg)',offset:.6},{transform:'perspective(700px) rotateY(-90deg)'}],{duration:260,easing:'cubic-bezier(.23,1,.32,1)',fill:'forwards'});const arrival={motion,cover,article:entry.target};mobileArrivals.add(arrival);const finish=()=>{cover.remove();mobileArrivals.delete(arrival);};motion.finished.then(finish).catch(finish);}}},{threshold:.15});chapters.forEach(c=>mobileObserver.observe(c));
    function cancelMobile(){for(const arrival of mobileArrivals){arrival.motion.cancel();arrival.cover.remove();}mobileArrivals.clear();}
    media.addEventListener('change',cancelMobile);$('story-motion').addEventListener('click',cancelMobile);
    document.addEventListener('keydown',()=>{cancelMobile();if(raf)settle();},true);
    new IntersectionObserver(entries=>{visible=entries[0].isIntersecting;if(!visible&&raf)settle();},{threshold:0}).observe(root);
    document.addEventListener('visibilitychange',()=>{if(document.hidden){cancelMobile();if(raf)settle();}});
    new ResizeObserver(()=>{if(raf)settle();else draw(canvas,step,current);mobileFrames();}).observe(canvas);
    window.addEventListener('beforeprint',()=>{cancelMobile();mobileFrames();settle();});
    const hash=()=>{const s=chapters.findIndex(c=>`#${c.id}`===location.hash);if(s>=0)activate(s,false);};window.addEventListener('hashchange',hash);
    canvas.addEventListener('click',event=>{if(raf||step!==0&&step!==4&&step!==2&&step!==3)return;const rect=canvas.getBoundingClientRect(),x=(event.clientX-rect.left)*1000/rect.width,y=(event.clientY-rect.top)*600/rect.height;let nearest=-1,distance=20;current.forEach((p,i)=>{if(p.opacity<.1)return;const d=Math.hypot(x-p.x,y-p.y);if(d<distance){distance=d;nearest=i;}});if(nearest>=0)api.select(nearest);});
    if(globalLayout){titles[2]='Полная сеть · признаки 2023';notes[2]=`${fmt(entities.length)} территорий, ${fmt(edges.length)} связей. Выделены ${graphList.length} аналогов. Раскладка условная; связи заданы в шести измерениях.`;const article=chapters[2];article.querySelector('p').textContent='Тот же муниципалитет в полной сети 2 016 территорий. Выделены его 15 ближайших аналогов по шести признакам расходов 2023 года.';article.querySelector('.story-note').textContent='PCA2 проецирует признаки на плоскость только для рисунка. Рёбра определены в шести измерениях; это не потоки.';}
    activate(0,false);hash();mobileFrames();hero();
    if($('hero-canvas'))new ResizeObserver(hero).observe($('hero-canvas'));
    // Inspection only: tests can verify state and identity without modifying science.
    window.__ATLAS_STORY__={get step(){return step;},get selectedId(){return entities[index].id;},get animating(){return !!raf;},get positions(){return current.map(p=>({id:p.id,x:p.x,y:p.y}));},get geometryIds(){return shapes.map(p=>p.id);},activate:s=>activate(s)};
    window.__ATLAS_STORY_METRICS__={initializationMs:performance.now()-initializedAt};
  }
})();
