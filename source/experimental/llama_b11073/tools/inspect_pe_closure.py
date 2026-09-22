#!/usr/bin/env python3
"""Recursive PE import-closure inspection for llama-server.exe (b11073 win-cuda-13.4).
Classifies every DLL in the closure:
  B = application-local, shipped in the release zip -> belongs in the delta pack
  A-SYSTEM  = Windows OS / UCRT / VC++ runtime (target always has these)
  A-NVIDIA  = CUDA runtime DLLs (target-provided per operator; NOT packaged)
"""
import subprocess, os, re, sys, json
from collections import deque

WORKDIR = os.environ.get("PE_CLOSURE_DIR", "/home/z/my-project/tmp-llama-pack/extracted")
START = "llama-server.exe"

SYSTEM_DLLS = {
    "KERNEL32.DLL","USER32.DLL","WS2_32.DLL","ADVAPI32.DLL","BCRYPT.DLL","SHCORE.DLL",
    "GDI32.DLL","SHELL32.DLL","OLE32.DLL","OLEAUT32.DLL","NTDLL.DLL","RPCRT4.DLL",
    "SECUR32.DLL","USERENV.DLL","WINMM.DLL","VERSION.DLL","MSVCRT.DLL","MSVCP140.DLL",
    "VCRUNTIME140.DLL","VCRUNTIME140_1.DLL","CONCRT140.DLL","VCOMP140.DLL",
    "UCRTBASE.DLL",
}
CRT_APISET_RE = re.compile(r"^API-MSCRT|API-MS-WIN-CRT", re.I)
NVIDIA_RE = re.compile(r"^(CUBLAS|CUBLASLT|CUDART|NVRTC|NVJITLINK|NVML|CUDA)\b|^(cublas|cublasLt|cudart|nvrtc|nvJitLink|nvml)", re.I)

def imports(pe_path):
    out = subprocess.run(["objdump","-p",pe_path], capture_output=True, text=True).stdout
    names = re.findall(r"^\s+DLL Name:\s+(\S+)$", out, re.M)
    return [n for n in names]

def classify(name, local_files):
    base = name.lower()
    if name.lower() in {f.lower() for f in local_files}:
        return "B"
    if base in SYSTEM_DLLS or CRT_APISET_RE.match(name):
        return "A-SYSTEM"
    if NVIDIA_RE.match(name):
        return "A-NVIDIA"
    return "UNKNOWN"

local_files = set(os.listdir(WORKDIR))
seen, edges, queue = {}, {}, deque([START])
while queue:
    mod = queue.popleft()
    if mod.lower() in seen: continue
    path = os.path.join(WORKDIR, mod)
    if not os.path.isfile(path):
        seen[mod.lower()] = classify(mod, local_files)
        edges[mod] = []
        continue
    imps = imports(path)
    edges[mod] = imps
    seen[mod.lower()] = "B"  # local + referenced
    for imp in imps:
        if imp.lower() not in seen:
            queue.append(imp)

report = {"closure_edges": edges, "classification": {}}
for mod, cat in sorted(seen.items()):
    report["classification"][mod] = cat

# Also inspect the dynamically-loadable backends (not in static closure):
DYNAMIC_BACKENDS = [f for f in sorted(local_files) if f.lower().startswith("ggml-") and f.lower().endswith(".dll")]
report["dynamic_backend_candidates"] = {}
for f in DYNAMIC_BACKENDS:
    report["dynamic_backend_candidates"][f] = imports(os.path.join(WORKDIR, f))

print(json.dumps(report, indent=1))
