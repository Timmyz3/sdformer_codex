from pathlib import Path
import sys

H = Path(__file__).resolve().parent
requested = sys.argv[1:] or ['small','held','disjoint','sequences']
for arm in ['moment','native_tap']:
    target = H/arm
    # verify_consumer imports only predict from the raw verifier under its H.
    if not (target/'verify_raw.py').exists():
        (target/'verify_raw.py').symlink_to(H.parent/'verify_raw.py')
    for kind in ['raw','consumer']:
        source = H.parent/f'verify_{kind}.py'
        code = source.read_text().replace('H=Path(__file__).resolve().parent','H=TARGET')
        sys.argv = [str(source),*requested]
        print(arm,kind,flush=True)
        exec(compile(code,str(source),'exec'),dict(__file__=str(source),TARGET=target))
