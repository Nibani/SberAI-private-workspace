function nearestRows(rows,count,compare){
 if(!Number.isSafeInteger(count)||count<0||rows.some(row=>Number.isNaN(row.d)))return rows.sort(compare).slice(0,count);
 count=Math.min(count,rows.length);if(count===0)return [];
 const best=[];
 for(const row of rows){
  if(best.length===count&&compare(row,best[best.length-1])>=0)continue;
  let low=0,high=best.length;
  while(low<high){const middle=(low+high)>>>1;if(compare(row,best[middle])<0)high=middle;else low=middle+1}
  best.splice(low,0,row);if(best.length>count)best.pop();
 }
 return best;
}
function sixNeighbors(index,excluded=-1){const a=D.entities[index],dims=[0,1,2,3,4,5].filter(j=>j!==excluded),distance=b=>Math.sqrt(dims.reduce((v,j)=>v+(a.v12_features[j]-b.v12_features[j])**2,0));if(excluded<0)return a.v12_neighbors.map(id=>({i:byId.get(id),d:distance(D.entities[byId.get(id)])}));return nearestRows(D.entities.map((b,i)=>({i,d:i===index?Infinity:distance(b)})),contest.v12.neighbors,(a,b)=>a.d-b.d||a.i-b.i)}
