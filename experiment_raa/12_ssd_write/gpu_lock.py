"""让实验服务持有启动器继承的同一个 WSL flock。"""

import errno
import fcntl
import os


def inherited_lock(path):
    """取与 path 指向同一 inode 的继承描述符，并检查它能持有独占锁。"""
    lock = path.stat()
    for name in os.listdir("/proc/self/fd"):
        fd = int(name)
        try:
            status = os.fstat(fd)
        except OSError as exc:
            # listdir 读取 /proc 时使用的描述符已由系统关闭。
            if exc.errno != errno.EBADF:
                raise
            continue
        if os.path.samestat(lock, status):
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
            return fd
    raise RuntimeError(f"实验须由 flock -x {path} 启动")
