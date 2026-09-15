#!/usr/bin/env python3
"""Install this additive kit only; never overwrite the existing robot nodes."""
import argparse,datetime,os,py_compile,shutil,tempfile
from pathlib import Path
from verify_compatibility import verify

def main():
    p=argparse.ArgumentParser();p.add_argument('--team',default='/opt/erc_ws/src/ku_sparcy_erc');a=p.parse_args()
    team=Path(a.team).resolve();verify(team)
    source=Path(__file__).resolve().parent;dest=team/'tools/day8_mesh_repair'
    dest.parent.mkdir(parents=True,exist_ok=True)
    temp=Path(tempfile.mkdtemp(prefix='.mesh_repair_install_',dir=dest.parent))
    try:
        for f in source.iterdir():
            if f.is_file() and f.suffix in ('.py','.sh','.json','.md'):shutil.copy2(f,temp/f.name)
        for f in temp.glob('*.py'):py_compile.compile(str(f),doraise=True)
        shutil.rmtree(temp/'__pycache__',ignore_errors=True)
        if dest.exists():
            backup=dest.with_name(dest.name+'.backup_'+datetime.datetime.now().strftime('%Y%m%d_%H%M%S_%f'))
            dest.rename(backup);print('Previous ADDITIVE kit saved:',backup)
        temp.rename(dest)
        for f in dest.glob('*.sh'):f.chmod(0o755)
        print('[ADDITIVE INSTALL][PASS]',dest)
        print('Existing nodes, configuration, URDF, models and controllers were NOT edited.')
    finally:
        if temp.exists():shutil.rmtree(temp)
if __name__=='__main__':main()
