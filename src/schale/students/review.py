"""Offline review UI: images, evidence, and explicit downloadable corrections."""

from pathlib import Path

from .models import Extraction


def write_review(
    result: Extraction, output: Path, filename: str = "review.html"
) -> None:
    payload = result.model_dump_json().replace("<", "\\u003c")
    html = r"""<!doctype html><meta charset="utf-8"><title>학생 정보 검토</title>
<style>body{font:15px system-ui;background:#f3f6fa;color:#172a40;margin:24px}header{position:sticky;top:0;background:#f3f6fa;padding:12px;z-index:2}button,select,input{font:inherit;padding:6px}article{background:white;margin:18px 0;padding:18px;border-radius:12px}img{max-width:100%;max-height:540px}table{border-collapse:collapse;width:100%}td,th{border-bottom:1px solid #ddd;text-align:left;padding:5px}input{width:100px}.unknown,.conflict{background:#fff0ce}.observed{background:#e9f1fb}details{font-size:13px}summary{cursor:pointer}.grid{display:grid;grid-template-columns:1fr 1fr;gap:20px}@media(max-width:1000px){.grid{display:block}}</style>
<style>table{table-layout:fixed}td{overflow-wrap:anywhere}td:nth-child(1){width:25%}td:nth-child(2){width:23%}td:nth-child(3){width:18%}td:nth-child(4){width:34%}td input{width:100%;box-sizing:border-box}canvas{max-width:100%;height:auto}</style>
<header><h1>학생 정보 검토</h1><p id="engine"></p><p id="stats"></p>
<label><input style="width:auto" id="filter" type="checkbox"> 검토가 필요한 학생만 표시 (미확인·충돌·단일 관측)</label>
<button id="save">수정사항 저장 (corrections.json)</button><p>confirmed: 여러 프레임 일치 · observed: 단일 관측 · inferred: 시각 상태 해석 · corrected: 원본 대조 후 수정 · conflict: 판독 충돌 · 빈칸: 확인 필요. 점수는 확률이 아닙니다.</p></header><main id="rows"></main>
<script>
const data=PAYLOAD;const edits={};
document.querySelector('#engine').textContent=`판독기: ${data.diagnostics.engine} · confirmed는 프레임 간 일치이며 정확도 보장이 아닙니다.`;
if(data.diagnostics.review_note)document.querySelector('#engine').textContent+=' '+data.diagnostics.review_note;
const names={bond:'인연',level:'레벨',star:'성급',ex:'EX',basic:'기본',passive:'강화',sub:'서브',weapon_star:'전용무기 성급',weapon_level:'전용무기 레벨',gear:'애용품 티어',equipment1:'장비 1 티어',equipment2:'장비 2 티어',equipment3:'장비 3 티어',equipment1_level:'장비 1 레벨',equipment2_level:'장비 2 레벨',equipment3_level:'장비 3 레벨',potential_hp:'능력 개방 체력',potential_attack:'능력 개방 공격',potential_heal:'능력 개방 치유'};
function el(tag,text){const n=document.createElement(tag);if(text!==undefined)n.textContent=text;return n;}
document.querySelector('#stats').textContent=`학생 ${data.students.length}명 · 식별 ${data.diagnostics.identified_students}명 · ${JSON.stringify(data.diagnostics.field_statuses)}`;
for(const s of data.students){
 const a=el('article');a.dataset.issue=(!['confirmed','corrected'].includes(s.identity_status)||Object.values(s.fields).some(f=>!['confirmed','corrected'].includes(f.status)))?'1':'0';
 a.append(el('h2',`${s.name} · ${s.student_id??'ID 미확인'} · ${s.identity_status}`));
 const label=el('label','학생 ID 수정: '),id=el('input');id.type='number';id.value=s.student_id??'';id.onchange=()=>{(edits[s.key]??={}).student_id=id.value===''?null:Number(id.value);};label.append(id);a.append(label);
 const grid=el('div');grid.className='grid';const visual=el('div'),img=el('img');img.loading='lazy';img.src=s.screenshots[0];const link=el('a');link.href=img.src;link.target='_blank';link.append(img);visual.append(link);
 if(s.screenshots.length>1){const select=el('select');for(const file of s.screenshots){const o=el('option',file);o.value=file;select.append(o);}select.onchange=()=>{img.src=select.value;link.href=select.value;};visual.append(select);}
 const table=el('table');for(const [key,f] of Object.entries(s.fields)){const tr=el('tr');tr.className=f.status;tr.append(el('td',names[key]??key));const cell=el('td'),input=el('input');input.type='number';input.min='0';input.value=f.value??'';input.placeholder='미확인';input.onchange=()=>{const e=edits[s.key]??={};(e.fields??={})[key]=input.value===''?null:Number(input.value);};cell.append(input);tr.append(cell,el('td',f.status));const d=el('details');d.append(el('summary','근거'));for(const e of f.evidence){d.append(el('p',`${e.timestamp.toFixed(3)}s · ${e.method} · ${e.raw}`));const c=el('canvas');const source=new Image();source.onload=()=>{const [x,y,x1,y1]=e.roi;c.width=Math.round((x1-x)*source.width);c.height=Math.round((y1-y)*source.height);c.getContext('2d').drawImage(source,x*source.width,y*source.height,c.width,c.height,0,0,c.width,c.height);};d.addEventListener('toggle',()=>{if(d.open&&!source.src)source.src=e.image;});d.append(c);}const evidence=el('td');evidence.append(d);tr.append(evidence);table.append(tr);}
 grid.append(visual,table);a.append(grid);document.querySelector('#rows').append(a);
}
document.querySelector('#filter').onchange=e=>document.querySelectorAll('article').forEach(a=>a.hidden=e.target.checked&&a.dataset.issue!=='1');
document.querySelector('#save').onclick=()=>{const blob=new Blob([JSON.stringify({schema_version:1,students:edits},null,2)],{type:'application/json'});const a=el('a');a.href=URL.createObjectURL(blob);a.download='corrections.json';a.click();setTimeout(()=>URL.revokeObjectURL(a.href),1000);};
</script>""".replace("PAYLOAD", payload)
    (output / filename).write_text(html, encoding="utf-8")
