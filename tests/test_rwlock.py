"""读写锁（utils.rwlock）行为测试 + 检索链路接线测试。

为什么这些用例重要：
1. 读写锁是「并发正确性」组件，错了不会报错、只会偶发卡死或读到脏数据，
   所以必须用真实线程去验证「读能并发 / 写能独占 / 写不被饿死」。
2. 接线用例守住「检索必须持读锁、重建索引必须持写锁」这个契约。
   没有它，将来有人把 retrieve_obsidian 改回裸调用，单测全绿、压测再次退化成队列。
"""

import threading
import time

import pytest

from knowledge import pipeline
from knowledge.config import config
from utils.rwlock import ReadWriteLock


class _Recorder:
    """记录 read() / write() 调用次数的代理，用来验证调用方持的是哪种锁。"""

    def __init__(self, inner):
        self._inner = inner
        self.calls: list[str] = []

    def read(self):
        self.calls.append("read")
        return self._inner.read()

    def write(self):
        self.calls.append("write")
        return self._inner.write()

    def stats(self):
        return self._inner.stats()


class _FakeStore:
    """最小向量库替身：空语料，不打开目录、不加载模型。"""

    def __init__(self, name: str, count: int = 0):
        self.collection_name = name
        self._count = count

    def search(self, query, k=5):
        return []

    def get(self, include=None):
        return {"documents": [], "metadatas": []}

    def count(self):
        return self._count


# ═══════════════════════════════════════════
#  一、锁本身的行为
# ═══════════════════════════════════════════

def test_readers_enter_concurrently():
    """4 个读线程必须能同时进入临界区；若实际被串行化，栅栏会超时。"""
    lock = ReadWriteLock()
    barrier = threading.Barrier(4, timeout=3)
    errors: list[BaseException] = []

    def reader():
        try:
            with lock.read():
                barrier.wait()  # 4 个都进来才能通过
        except BaseException as e:  # noqa: BLE001 - 线程内异常需带回主线程
            errors.append(e)

    threads = [threading.Thread(target=reader) for _ in range(4)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=5)
    assert not errors, f"读并发失败: {errors}"
    assert lock.stats()["readers"] == 0


def test_read_lock_is_reentrant_for_same_thread():
    """同一线程重复获取读锁不能自锁（query_obsidian 有嵌套读的写法）。"""
    lock = ReadWriteLock()
    with lock.read(), lock.read():
        assert lock.stats()["readers"] == 1  # 同一线程只算一个读者
    assert lock.stats()["readers"] == 0


def test_write_is_exclusive():
    """写锁持有时，后来的读者必须等待。"""
    lock = ReadWriteLock()
    order: list[str] = []
    write_holding = threading.Event()

    def writer():
        with lock.write():
            order.append("write-enter")
            write_holding.set()
            time.sleep(0.2)
            order.append("write-exit")

    t = threading.Thread(target=writer)
    t.start()
    assert write_holding.wait(2)

    with lock.read():
        order.append("read-enter")
    t.join(timeout=5)

    # 读者只能在写者释放后进入
    assert order.index("write-exit") < order.index("read-enter"), order


def test_new_readers_yield_to_waiting_writer():
    """写者排队时，新读者要让路——否则重建索引会被持续查询饿死。"""
    lock = ReadWriteLock()
    reader_holding = threading.Event()
    release_reader = threading.Event()
    reader2_locked = threading.Event()
    order: list[str] = []

    def first_reader():
        with lock.read():
            reader_holding.set()
            release_reader.wait(3)

    def writer():
        reader_holding.wait(3)
        order.append("writer-wait-start")
        with lock.write():
            order.append("writer-enter")
            time.sleep(0.15)

    def second_reader():
        reader_holding.wait(3)
        time.sleep(0.05)  # 保证写者已经排在队列里
        with lock.read():
            order.append("reader2-enter")
        reader2_locked.set()

    t1 = threading.Thread(target=first_reader)
    t2 = threading.Thread(target=writer)
    t3 = threading.Thread(target=second_reader)
    for t in (t1, t2, t3):
        t.start()

    time.sleep(0.5)
    release_reader.set()
    for t in (t1, t2, t3):
        t.join(timeout=5)

    assert reader2_locked.is_set(), "第二个读者没有被唤醒"
    # 写者必须在第二个读者之前拿到锁
    assert order.index("writer-enter") < order.index("reader2-enter"), order


def test_exception_inside_lock_releases_it():
    """临界区抛异常时必须释放锁，否则一次业务异常就会永久锁死索引。"""
    lock = ReadWriteLock()
    with pytest.raises(ValueError), lock.write():
        raise ValueError("业务异常")

    assert lock.stats() == {"readers": 0, "writer_active": False, "writers_waiting": 0}
    # 锁已释放：另一个线程能拿到写锁
    acquired = threading.Event()

    def worker():
        with lock.write():
            acquired.set()

    t = threading.Thread(target=worker)
    t.start()
    t.join(timeout=3)
    assert acquired.is_set()


def test_mixed_nesting_is_rejected_not_deadlocked():
    """持读锁再求写锁必须立刻报错，不能默默死锁。"""
    lock = ReadWriteLock()
    with lock.read(), pytest.raises(RuntimeError, match="已持有读锁"), lock.write():
        pass

    with lock.write(), pytest.raises(RuntimeError, match="已持有写锁"), lock.read():
        pass


# ═══════════════════════════════════════════
#  二、检索链路的接线契约
# ═══════════════════════════════════════════

def test_retrieve_takes_read_lock(monkeypatch):
    """retrieve_obsidian 必须持读锁：并发检索不能被串行化。"""
    recorder = _Recorder(ReadWriteLock())
    monkeypatch.setattr(pipeline, "_obsidian_rwlock", recorder)
    monkeypatch.setattr(pipeline, "get_vector_store", lambda name=None: _FakeStore("rwlock_read"))
    monkeypatch.setattr(config, "RERANK_ENABLED", False)

    pipeline.retrieve_obsidian("任意问题", 3)

    assert recorder.calls == ["read"], f"实际持锁方式: {recorder.calls}"


def test_index_takes_write_lock(monkeypatch, tmp_path):
    """index_obsidian 必须持写锁：重建期间不允许任何检索读到半成品索引。"""
    recorder = _Recorder(ReadWriteLock())
    monkeypatch.setattr(pipeline, "_obsidian_rwlock", recorder)
    # vault 置空 → 实现会立刻返回 error，但持锁动作已经发生
    monkeypatch.setattr(pipeline, "OBSIDIAN_VAULT_DIR", "")

    result = pipeline.index_obsidian()

    assert result["status"] == "error"
    assert recorder.calls == ["write"], f"实际持锁方式: {recorder.calls}"
