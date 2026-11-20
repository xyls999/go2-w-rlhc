"""以完全脱离控制台的方式启动训练（避免父控制台关闭时子进程被 abort）。"""
import subprocess
import sys

DETACHED = 0x00000008 | 0x00000200  # DETACHED_PROCESS | CREATE_NEW_PROCESS_GROUP

cmd = sys.argv[1:]
out = None
for i, a in enumerate(cmd):
    if a == "--out":
        out = cmd[i + 1]
assert out, "need --out"
log = open(out + r"\train.log", "ab", buffering=0)
err = open(out + r"\train.err.log", "ab", buffering=0)
p = subprocess.Popen(cmd, stdout=log, stderr=err, stdin=subprocess.DEVNULL,
                     creationflags=DETACHED, close_fds=True)
print("detached pid", p.pid)
