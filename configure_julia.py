import subprocess
import re
import sys
import os.path
import julia

julia.install()

pathrgx = re.compile(b"Path:[\t ]*([^\r\n]*)")
envinfo = subprocess.check_output("poetry env info", shell=True)
envdir = str(pathrgx.findall(envinfo)[0], "utf")

envpath = os.path.join(envdir, "Scripts", "python.exe" if sys.platform == "win32" else "python")

print(f"Path to environment interpreter: '{envpath}'")

fenvpath = envpath.replace("\\", "\\\\")

rebuild_pycall = f"""
import Pkg
ENV["PYTHON"]="{fenvpath}"
Pkg.add(["Distributions", "NLopt"])
Pkg.build("PyCall")
""".splitlines()

cmd = ["julia"] + sum((['--eval', x] for x in rebuild_pycall if len(x) > 0), [])
print(cmd)
ret = subprocess.call(cmd)