// Exercise all 24 predeclared report selections without relying on a browser.
const fs=require('fs'),vm=require('vm'),assert=require('assert');
const path=require('path'),base=path.resolve(__dirname,'..');
const html=fs.readFileSync(path.join(base,'report.html'),'utf8');
const match=html.match(/<script type="application\/json" id="snapshot">([\s\S]*?)<\/script>/);
assert(match);
const data=JSON.parse(match[1]);
const nodes={};
function node(id){return nodes[id]||(nodes[id]={value:'',textContent:'',style:{},addEventListener(){}});}
node('snapshot').textContent=match[1];node('layer').value='0';node('block').value='32';node('keep').value='4';
const sandbox={document:{getElementById:node}};
vm.createContext(sandbox);
vm.runInContext(fs.readFileSync(path.join(base,'scripts/report-interaction.js'),'utf8'),sandbox);
let checked=0;
for(let l=0;l<2;l++)for(const B of [16,32,64])for(const K of [0,2,4,8]){
 node('layer').value=String(l);node('block').value=String(B);node('keep').value=String(K);
 vm.runInContext('update()',sandbox);
 const r=data[l].rows.find(x=>x.B===B&&x.K===K);
 const expected=[r.packet_failure_fraction,r.replay_FC1_fraction_by_h,r.replay_FC1_fraction_fixed96];
 expected.forEach((x,i)=>{assert(Math.abs(parseFloat(node('bar'+i).style.width)-100*x)<1e-10);assert(node('value'+i).textContent.endsWith('%'));});
 for(const id of ['identity','packets','wide','source-read','changes'])assert(node(id).textContent.length>0);
 assert(!Object.values(nodes).some(x=>/NaN|undefined|Infinity/.test(x.textContent)));
 checked++;
}
console.log(JSON.stringify({status:'PASS',selections:checked,scope:'DOM-adapter interaction/data binding, not visual browser QA'}));
