
const rows=JSON.parse(document.getElementById('snapshot').textContent);
const fmt=(x,d=4)=>Number(x).toLocaleString('zh-CN',{minimumFractionDigits:d,maximumFractionDigits:d});
function update(){
 const l=rows[Number(document.getElementById('layer').value)],B=Number(document.getElementById('block').value),K=Number(document.getElementById('keep').value);
 const r=l.rows.find(x=>x.B===B&&x.K===K);
 const metrics=[r.packet_failure_fraction,r.replay_FC1_fraction_by_h,r.replay_FC1_fraction_fixed96];
 metrics.forEach((v,i)=>{document.getElementById('bar'+i).style.width=(100*v)+'%';document.getElementById('value'+i).textContent=fmt(100*v)+'%';});
 document.getElementById('packets').textContent=fmt(r.bare_payload_bytes_by_assumed_U_width['32']/1e6,3)+' MB';
 document.getElementById('wide').textContent=fmt(l.N_includes_T*l.H*4/1e6,3)+' MB';
 document.getElementById('source-read').textContent=fmt(r.replay_PSN_terms_by_h*l.C/(8*l.T)/1e6,3)+' MB';
 document.getElementById('changes').textContent=r.certified_patch_bits.toLocaleString('zh-CN')+' 位';
 document.getElementById('identity').textContent='sample0 · '+l.module.split('layers.')[1]+' · N='+l.N_includes_T+'（含 T=10）· C='+l.C+' · H='+l.H;
}
['layer','block','keep'].forEach(id=>document.getElementById(id).addEventListener('change',update));update();
