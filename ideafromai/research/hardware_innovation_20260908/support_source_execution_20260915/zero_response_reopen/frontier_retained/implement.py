"""Copy frozen ordinary strong control; only select Wz zero-aware backend and both strong arms."""
from pathlib import Path
HERE=Path(__file__).resolve().parent
ZERO=HERE.parent
SOURCE=ZERO.parent
OLD=SOURCE/'frontier_joined/retained_config'
s=(OLD/'joined_core.sv').read_text()
assert s.count('support_fc1 backend_core(')==1
(HERE/'joined_core.sv').write_text(s.replace('support_fc1 backend_core(', 'support_fc1_zero backend_core('))
(HERE/'frontier_source.sv').write_text((OLD/'frontier_source.sv').read_text())
(HERE/'support_fc1_zero.sv').write_text((ZERO/'support_fc1_zero.sv').read_text())
s=(OLD/'tb.cpp').read_text()
assert s.count('for(int j=1;j<k;j++)')==1
s=s.replace('for(int j=1;j<k;j++)','for(int j=0;j<k;j++)')
a='for(int frontier=0;frontier<(mode<2?1:2);frontier++)'
assert s.count(a)==1
s=s.replace(a,'for(int frontier=1;frontier<2;frontier++)')
s=s.replace('"graph per-P configuration"','"graph P0 cold, P1..31 retained configuration"')
(HERE/'tb.cpp').write_text(s)

# Reuse the already checked Wz static exporter, changing only source scope/path.
s=(ZERO/'prepare_joined.py').read_text()
s=s.replace('SOURCE=HERE.parent','ZERO=HERE.parent\nSOURCE=ZERO.parent')
s=s.replace("src=np.load(SOURCE/'source_cases.npz')", "src=np.load(SOURCE/'source_class_adapt/expanded_sources/source_cases.npz')")
s=s.replace("adapted=np.load(HERE/", "adapted=np.load(ZERO/")
s=s.replace("Wp=np.load(HERE/", "Wp=np.load(ZERO/")
s=s.replace('for i in range(2):cases.append', 'for i in range(len(src[\'X_q16\'])):cases.append')
s=s.replace("stem='joined_'+args.variant", "stem='inputs_expanded'")
s=s.replace("str(SOURCE/'source_cases.npz')", "str(SOURCE/'source_class_adapt/expanded_sources/source_cases.npz')")
s=s.replace('first two training frames, same 32 sampled positions; not held out', '32 fixed training frames, P32 each; only frame0 selected Wz; no held-out AEE')
(HERE/'prepare.py').write_text(s)
