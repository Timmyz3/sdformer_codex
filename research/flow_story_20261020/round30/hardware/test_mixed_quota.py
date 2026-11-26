"""Independent address/row/contract checks, no model or GPU execution."""
import importlib.util, json, sys, unittest
from pathlib import Path
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE))
import mixed_quota as m
CONTRACT=HERE/'MASK_CONTRACT_V1.json'
RESULT=HERE/'MIXED_COST_V1.json'

class Tests(unittest.TestCase):
 def setUp(self):
  self.c=json.loads(CONTRACT.read_text());self.r=json.loads(RESULT.read_text())
 def test_production_contract(self):
  implementation=HERE.parent/'implementation'
  sys.path.insert(0,str(implementation))
  import mask_contract
  self.assertIs(mask_contract.validate(self.c),self.c)
  for step in (1,256,512):
   coords=mask_contract.coordinates(self.c,'crop_center9',step=step)
   for g,v in self.c['geometries']['crop_center9'].items():
    self.assertEqual(len(coords[g]),v['Q8_complete_tiles'])
    self.assertEqual(coords,mask_contract.coordinates(self.c,'crop_center9',step=step))
 def test_each_pixel_partition(self):
  for geometry,groups in self.c['geometries'].items():
   for name,row in groups.items():
    h,w=row['H'],row['W'];k=row['Q8_complete_tiles']
    selected={tuple(x) for x in self.c['evaluation_tiles'][name]} if geometry=='full9' else {(i//(w//9),i%(w//9)) for i in range(k)}
    counts={'tail':0,'restored':0,'low':0}
    for y in range(h):
     for x in range(w):
      key='tail' if y//9>=h//9 or x//9>=w//9 else 'restored' if (y//9,x//9) in selected else 'low'
      counts[key]+=1
    self.assertEqual(sum(counts.values()),h*w)
    self.assertEqual(counts['restored'],81*k)
    self.assertEqual(counts['tail'],h*w-(h//9*9)*(w//9*9))
    self.assertEqual(10*(8*(counts['tail']+counts['restored'])+4*counts['low']),10*8*h*w-40*counts['low'])
 def test_actual_job_sums_and_gate(self):
  summed={k:self.r['common'][k] for k in m.KEYS}
  for rec in self.r['actual']['records']:
   for k in m.KEYS:summed[k]+=rec['core'].get(k,0)+rec['mixed_obligations'].get(k,0)
   self.assertEqual(rec['row_planes'],rec['rows']*4+rec['high8_rows']*4)
   self.assertEqual(rec['restored_rows'],810*rec['Q8_complete_tiles'])
  self.assertEqual(summed,self.r['actual']['summary'])
  pe=summed['PE_arithmetic_ticks']
  reserve=m.fractions.Fraction(m.RESERVE_NUM,m.RESERVE_DEN)
  for baseline in self.r['ordinary'].values():
   self.assertLessEqual(pe+reserve,m.fractions.Fraction(9,10)*baseline['PE_arithmetic_ticks'])
   self.assertLessEqual(summed['PE_reserved_ticks']+reserve,m.fractions.Fraction(9,10)*baseline['PE_reserved_ticks'])
   self.assertLessEqual(summed['ticks'],m.fractions.Fraction(92,100)*baseline['ticks'])
 def test_maximum_discrete_rule(self):
  rows=self.r['screening'];i=self.r['selected_breakpoint']
  self.assertTrue(rows[i]['passes_reserve'])
  self.assertTrue(all(not x['passes_reserve'] for x in rows[i+1:]))
  f=self.c['common_fraction']
  for geom in self.c['geometries'].values():
   for v in geom.values():self.assertEqual(v['Q8_complete_tiles'],f['numerator']*v['complete_tiles']//f['denominator'])
 def test_true_tail_and_fc_pair(self):
  records=self.r['actual']['records']
  mlp={}
  for rec in records:
   if '.mlp.' in rec['name']:mlp.setdefault(rec['group'],[]).append(rec)
  self.assertEqual(len(mlp),12)
  for rows in mlp.values():
   self.assertEqual(len(rows),2)
   self.assertEqual(rows[0]['high8_rows'],rows[1]['high8_rows'])
   self.assertGreater(rows[0]['tail_rows'],0)
 def test_source_immutable_and_staging_capacity(self):
  self.assertEqual(m.s.sha(m.OLD/'unified_service.py'),m.SOURCE_SHA)
  machine=m.s.Machine()
  for rec in self.r['actual']['records']:
   o=rec['mixed_obligations']
   self.assertLessEqual(o['bitmap_storage_bytes']+64,2048)
   self.assertGreaterEqual(o['staging_port_ticks'],m.s.ceil(o['local_input_bytes'],machine.input_port_bytes_tick)+m.s.ceil(o['local_output_bytes'],machine.output_port_bytes_tick))
   self.assertEqual(o['scatter_DMA_startup_ticks'],o['scatter_spans']*32)

if __name__=='__main__':unittest.main()
