# -*- coding: utf-8 -*-
"""Zip the code used for the 036 reconstructions (three pipelines with the dark-field and partial-coherence extensions, d036 scripts, the
comparison script and the d018 tools used for 036).  usage (from the workspace root): python d036/make_pipeline_zip_036.py"""
import glob, os, sys, zipfile
NAME = "FPT036_暗視野対応パイプライン_20260926"
OUT = sys.argv[1] if len(sys.argv) > 1 else f"d036/{NAME}.zip"
PATS = ["README_036_暗視野パイプライン.md", "build_pptx.py",
        "fpt_pipeline/README.md", "fpt_pipeline/environment.yml", "fpt_pipeline/run_pipeline.py", "fpt_pipeline/config_*.json",
        "fpt_pipeline/fptrecon/*.py",
        "dip_pipeline/README.md", "dip_pipeline/environment_dip.yml", "dip_pipeline/run_dip.py", "dip_pipeline/config_dip_*.json",
        "dip_pipeline/dipfpm/*.py", "dip_pipeline/tools/*.py",
        "epry_pipeline/README.md", "epry_pipeline/environment_epry.yml", "epry_pipeline/run_epry.py", "epry_pipeline/config_epry_*.json",
        "epry_pipeline/eprfpm/*.py",
        "d036/*.py", "d036/*.sh", "d036/tools/*.py", "d036/pc/*.py", "d036/pc/*.sh",
        "d018/compare_018.py", "d018/tools/blis_objband.py", "d018/tools/kappa_scan.py", "d018/tools/blis_bandcheck.py"]
files = sorted({p for pat in PATS for p in glob.glob(pat)} - {"d036/make_pipeline_zip_036.py"}) + ["d036/make_pipeline_zip_036.py"]
missing = [pat for pat in PATS if not glob.glob(pat)]
assert not missing, missing
with zipfile.ZipFile(OUT, "w", zipfile.ZIP_DEFLATED) as z:
    for p in files:
        z.write(p, f"{NAME}/{p}")
print(f"{len(files)} files -> {OUT} ({os.path.getsize(OUT) / 1e3:.0f} kB)")
