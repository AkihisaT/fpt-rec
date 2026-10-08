"""Quick self-test of the FPT pipeline repository (no raw data needed, a few tens of seconds).

usage (repository root, environment 'dip'):  python tests/selftest.py
Checks: every .py compiles, every .json parses, every .sh passes `bash -n`, all fpt_pipeline configs load (raw-data
variables replaced by a dummy path when not set), the three packages import, the posaffine sign self-test passes,
and no computer-specific absolute path (/Users/..., /home/..., C:\\Users\\...) is left in the text files that Git
tracks or would add (outputs excluded by .gitignore, e.g. dip_output*/config_used.json, may contain such paths).
Exit code 0 = all passed."""
import glob, json, os, py_compile, re, subprocess, sys, tempfile
R = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) + "/"
os.chdir(R)
res = {}
files = lambda pat: sorted(f for f in glob.glob(pat, recursive=True) if "__pycache__" not in f)
bad = []
for f in files("**/*.py"):
    try: py_compile.compile(f, doraise=True, cfile=os.path.join(tempfile.gettempdir(), "fpt_selftest.pyc"))
    except Exception as e: bad.append(f)
res["python files compile"] = (not bad, f"{len(files('**/*.py'))} files" + (f", errors {bad}" if bad else ""))
dot = lambda pat: [f for f in files(".claude/" + pat) if not f.startswith(".claude/worktrees/")]   # hidden folder, without
bad = []                                                                                           # the app's worktrees
for f in files("**/*.json") + dot("**/*.json"):
    try: json.load(open(f, encoding="utf-8"))
    except Exception: bad.append(f)
res["json files parse"] = (not bad, f"{len(files('**/*.json') + dot('**/*.json'))} files" + (f", errors {bad}" if bad else ""))
sh = files("**/*.sh") + dot("**/*.sh")
bad = [f for f in sh if subprocess.run(["bash", "-n", f], capture_output=True).returncode]
res["shell scripts syntax"] = (not bad, f"{len(sh)} files" + (f", errors {bad}" if bad else ""))
for v in ("FPT003_RAW", "FPT018_RAW", "FPT036_RAW", "FPT32A_RAW"):
    os.environ.setdefault(v, "/nonexistent")
sys.path.insert(0, R + "fpt_pipeline")
try:
    from fptrecon.config import load_config
    cf = files("fpt_pipeline/config_*.json"); [load_config(f) for f in cf]
    res["fpt_pipeline configs load"] = (True, f"{len(cf)} configs")
except Exception as e:
    res["fpt_pipeline configs load"] = (False, repr(e)[:200])
try:
    from fptrecon import posaffine, coherence, nonlinear                    # noqa: F401
    ok = posaffine.selftest_sign()
    ok = ok[0] if isinstance(ok, tuple) else bool(ok)
    res["fptrecon import + posaffine sign self-test"] = (bool(ok), "")
except Exception as e:
    res["fptrecon import + posaffine sign self-test"] = (False, repr(e)[:200])
for name, d, mods in (("dipfpm", "dip_pipeline", ("dipfpm.coherence", "dipfpm.physics")), ("eprfpm", "epry_pipeline", ("eprfpm.core",))):
    r = subprocess.run([sys.executable, "-c", "import importlib, sys; sys.path.insert(0, '.'); sys.path.insert(0, '../dip_pipeline'); "
                        + "; ".join(f"importlib.import_module('{m}')" for m in mods)], cwd=R + d, capture_output=True, text=True)
    res[f"{name} import"] = (r.returncode == 0, r.stderr.strip().splitlines()[-1] if r.returncode else "")
try:                                     # verdicts of tests/compare_ref.py on copies of the reference changed by hand
    sys.path.insert(0, R + "tests"); import compare_ref as cref
    ref = cref.load_ref()
    def changed(defocus=0.0, astig=0.0, corr=0.0, misfit=1.0, edit=None):
        pa, cal = json.loads(json.dumps(ref[0])), json.loads(json.dumps(ref[1]))
        cal["defocus_um"] += defocus; cal["astig_um"][0] += astig; pa["correction_px"][0][0] += corr
        pa["aligned_stack"]["misfit"] *= misfit
        if edit: edit(pa, cal)
        return pa, cal
    near = dict(defocus=20.0, astig=20.0, corr=0.05, misfit=1.01)          # within the tolerances for another computer
    cases = (("一致", changed()), ("ほぼ一致", changed(defocus=1.0)), ("ほぼ一致", changed(**near)),
             ("不一致", changed(defocus=500.0)), ("不一致", changed(astig=500.0)), ("不一致", changed(corr=0.5)),
             ("不一致", changed(misfit=1.05)),
             ("不一致", changed(**near, edit=lambda pa, cal: cal.__setitem__("kc", cal["kc"] * 1.1))),
             ("不一致", changed(**near, edit=lambda pa, cal: pa.__setitem__("consistent", not pa["consistent"]))),
             ("不一致", changed(**near, edit=lambda pa, cal: pa.pop("check"))))
    got = [cref.compare(r, ref)[0] for _, r in cases]
    ok = all(g.startswith(w) for g, (w, _) in zip(got, cases))
    res["compare_ref verdicts"] = (ok, f"{len(cases)} cases" + ("" if ok else f", got {got}"))
except Exception as e:
    res["compare_ref verdicts"] = (False, repr(e)[:200])
try:                                     # session-start hook on a computer without conda and paths_local.sh
    with tempfile.TemporaryDirectory() as t:
        env = {"HOME": t, "PATH": "/usr/bin:/bin", "CLAUDE_PROJECT_DIR": t, "CLAUDE_ENV_FILE": os.path.join(t, "env")}
        r = subprocess.run(["bash", R + ".claude/hooks/fpt_session_env.sh"], env=env, capture_output=True, text=True, timeout=60)
        written = open(env["CLAUDE_ENV_FILE"]).read() if os.path.exists(env["CLAUDE_ENV_FILE"]) else ""
    ok = r.returncode == 0 and "/fpt-setup" in r.stdout and "conda activate" not in written
    res["session hook without conda"] = (ok, "" if ok else f"exit {r.returncode}: {r.stdout.strip()[:150]}")
except Exception as e:
    res["session hook without conda"] = (False, repr(e)[:200])
pat = re.compile(r"(/Users/[A-Za-z0-9_.-]+/|/home/[A-Za-z0-9_.-]+/|[A-Za-z]:\\\\Users\\\\)")
try:                                     # files Git tracks or would add; not the ignored outputs
    g = subprocess.run(["git", "ls-files", "-z", "--cached", "--others", "--exclude-standard"], cwd=R, capture_output=True, text=True)
    text_files = [f for f in g.stdout.split("\0") if f] if g.returncode == 0 else None
except OSError: text_files = None
if text_files is None: text_files = files("**/*.*") + dot("**/*.*")       # not a Git checkout (e.g. a zip): every file
bad = []
for f in text_files:
    if f.endswith((".py", ".sh", ".json", ".md", ".yml", ".txt")) and f != "tests/selftest.py" and os.path.isfile(f):
        if pat.search(open(f, encoding="utf-8", errors="ignore").read()): bad.append(f)
res["no computer-specific absolute paths"] = (not bad, ", ".join(bad))
w = max(len(k) for k in res)
for k, (ok, note) in res.items():
    print(f"{'OK ' if ok else 'NG '} {k:<{w}}  {note}")
sys.exit(0 if all(ok for ok, _ in res.values()) else 1)
