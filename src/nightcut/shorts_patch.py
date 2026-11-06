from pathlib import Path
import shutil
p=Path(r'C:\Users\Timo\Desktop\nightcut\src\nightcut\shorts.py')
# This is a patch script, not a replacement for shorts.py.
s=p.read_text(encoding='utf-8')
old='''    if lay.get("facecam"):
'''
new='''    # A portrait VOD with a burned-in facecam already has its final layout.
    # Preserve the full image instead of cropping and stacking it again.
    if info.width * 16 == info.height * 9 and lay.get("facecam_burned_in", False):
        graph = f"[0:v]scale={W}:{H},setsar=1[stack];"
        src = "[stack]"
    elif lay.get("facecam"):
'''
if old not in s:
    raise SystemExit('Patch failed: expected layout branch not found; no changes made.')
s=s.replace(old,new,1)
old_caption='''    if cfg["layout"].get("facecam"):
'''
new_caption='''    # On a precomposed portrait VOD, keep captions in the lower-middle area.
    if cfg["layout"].get("facecam_burned_in", False) and cfg["layout"].get("gameplay", {}).get("h") == 1.0:
        return int(H * 0.7)
    if cfg["layout"].get("facecam"):
'''
# Do not change caption position automatically: it's user-specific.
backup=p.with_suffix('.py.bak')
shutil.copy2(p, backup)
p.write_text(s,encoding='utf-8')
print('Backup:', backup)
print('Patch applied:',p)
