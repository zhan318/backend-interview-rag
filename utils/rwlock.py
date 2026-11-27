"""读写锁：读并发、写独占、写优先。

适用场景是「读多写极少」的共享资源。典型例子是向量库索引：
检索（读）每轮对话都会发生，重建索引（写）一天可能只有一次。
用一把普通互斥锁时，N 个并发检索会被排成队列，延迟随并发数线性放大。

为什么不引第三方库：实现只有几十行，行为可单测、可控，少一个依赖。

两类自锁（同一个线程重复获取不兼容的锁）会被显式检测并抛 RuntimeError，
而不是默默挂住——死锁在现场几乎无法定位，早失败远好于静默卡死。
"""

from __future__ import annotations

import threading
from contextlib import contextmanager


class ReadWriteLock:
    """写优先的读写锁。

    规则：
    - 读锁：多个线程可同时持有；同一线程重复获取只累加计数（读锁可重入）
    - 写锁：独占；有写者排队时，新读者要让路（否则重建索引会被持续查询饿死）
    - 同一个线程不允许「持读锁再求写锁」或「持写锁再求读锁」
    """

    def __init__(self) -> None:
        self._cond = threading.Condition(threading.Lock())
        self._readers: dict[int, int] = {}  # 线程 id → 重入计数
        self._writer: int | None = None  # 持有写锁的线程 id
        self._writers_waiting = 0

    # ── 读 ──

    @contextmanager
    def read(self):
        tid = threading.get_ident()
        with self._cond:
            if self._writer == tid:
                raise RuntimeError("当前线程已持有写锁，不能再获取读锁（会自锁）")
            if tid in self._readers:
                # 重入：不等待。否则写者在排队时，同一线程的第二次读会自己把自己挡死
                self._readers[tid] += 1
            else:
                while self._writer is not None or self._writers_waiting:
                    self._cond.wait()
                self._readers[tid] = 1
        try:
            yield
        finally:
            with self._cond:
                left = self._readers.get(tid, 0) - 1
                if left > 0:
                    self._readers[tid] = left
                else:
                    self._readers.pop(tid, None)
                    if not self._readers:
                        self._cond.notify_all()

    # ── 写 ──

    @contextmanager
    def write(self):
        tid = threading.get_ident()
        with self._cond:
            if self._writer == tid:
                raise RuntimeError("写锁不可重入")
            if tid in self._readers:
                raise RuntimeError("当前线程已持有读锁，不能再获取写锁（会自锁）")
            self._writers_waiting += 1
            try:
                while self._writer is not None or self._readers:
                    self._cond.wait()
            finally:
                self._writers_waiting -= 1
            self._writer = tid
        try:
            yield
        finally:
            with self._cond:
                self._writer = None
                self._cond.notify_all()

    # ── 观测 ──

    def stats(self) -> dict:
        """当前持锁状态，供压测与排障读取（不阻塞）。"""
        with self._cond:
            return {
                "readers": len(self._readers),
                "writer_active": self._writer is not None,
                "writers_waiting": self._writers_waiting,
            }
