import threading
import unittest

from cachetools import LeaseCache, LRUCache, TTLCache


class Timer:
    """Controllable timer, same style as tests/test_ttl.py."""

    def __init__(self, auto=False):
        self.auto = auto
        self.time = 0

    def __call__(self):
        if self.auto:
            self.time += 1
        return self.time

    def tick(self):
        self.time += 1


class Recorder:
    """Cleanup callback recording (key, value) calls."""

    def __init__(self):
        self.calls = []

    def __call__(self, key, value):
        self.calls.append((key, value))


class LeaseCacheTest(unittest.TestCase):
    def test_lease_basic(self):
        cleanup = Recorder()
        cache = LeaseCache(LRUCache(maxsize=10), cleanup=cleanup)
        cache["k"] = "v1"
        lease = cache.lease("k")
        self.assertEqual("v1", lease.value)
        self.assertEqual("k", lease.key)
        self.assertEqual(0, lease.generation)
        self.assertFalse(lease.released)
        lease.release()
        self.assertTrue(lease.released)
        # value still cached: no cleanup on release
        self.assertEqual([], cleanup.calls)
        self.assertEqual("v1", cache["k"])

    def test_get_lease_returns_value_and_lease(self):
        cache = LeaseCache(LRUCache(maxsize=10))
        cache["k"] = "v1"
        value, lease = cache.get_lease("k")
        self.assertEqual("v1", value)
        self.assertEqual("v1", lease.value)
        lease.release()

    def test_lease_missing_key(self):
        cache = LeaseCache(LRUCache(maxsize=10))
        with self.assertRaises(KeyError):
            cache.lease("missing")
        with self.assertRaises(KeyError):
            cache.get_lease("missing")

    def test_lease_context_manager(self):
        cleanup = Recorder()
        cache = LeaseCache(LRUCache(maxsize=10), cleanup=cleanup)
        cache["k"] = "v1"
        with cache.lease("k") as lease:
            self.assertEqual("v1", lease.value)
            del cache["k"]
        # cleanup runs when the with-block releases the last lease
        self.assertEqual([("k", "v1")], cleanup.calls)

    def test_delete_rewrite_release_cleans_only_old_generation(self):
        # acceptance scenario: lease v1, delete, write v2, release v1
        cleanup = Recorder()
        cache = LeaseCache(LRUCache(maxsize=10), cleanup=cleanup)
        cache["k"] = "v1"
        lease = cache.lease("k")
        del cache["k"]
        self.assertEqual([], cleanup.calls)  # lease still active
        cache["k"] = "v2"
        lease.release()
        # only v1 is cleaned up, v2 is untouched and still readable
        self.assertEqual([("k", "v1")], cleanup.calls)
        self.assertEqual("v2", cache["k"])
        self.assertEqual("v2", cache.lease("k").value)

    def test_last_release_cleans_up(self):
        cleanup = Recorder()
        cache = LeaseCache(LRUCache(maxsize=10), cleanup=cleanup)
        cache["k"] = "v1"
        lease1 = cache.lease("k")
        lease2 = cache.lease("k")
        del cache["k"]
        self.assertEqual([], cleanup.calls)
        lease1.release()
        self.assertEqual([], cleanup.calls)  # lease2 still active
        lease2.release()
        self.assertEqual([("k", "v1")], cleanup.calls)

    def test_no_new_lease_after_delete(self):
        cache = LeaseCache(LRUCache(maxsize=10))
        cache["k"] = "v1"
        lease = cache.lease("k")
        del cache["k"]
        with self.assertRaises(KeyError):
            cache.lease("k")
        # existing lease remains usable until released
        self.assertEqual("v1", lease.value)
        lease.release()

    def test_overwrite_generations(self):
        cleanup = Recorder()
        cache = LeaseCache(LRUCache(maxsize=10), cleanup=cleanup)
        cache["k"] = "v1"
        lease1 = cache.lease("k")
        cache["k"] = "v2"  # new generation, v1 retired but leased
        self.assertEqual([], cleanup.calls)
        self.assertEqual(0, lease1.generation)
        lease2 = cache.lease("k")
        self.assertEqual(1, lease2.generation)
        self.assertEqual("v2", lease2.value)
        lease1.release()  # old lease must not clean up the new value
        self.assertEqual([("k", "v1")], cleanup.calls)
        self.assertEqual("v2", cache["k"])
        lease2.release()
        self.assertEqual([("k", "v1")], cleanup.calls)  # v2 still cached

    def test_expired_item_no_new_lease(self):
        cleanup = Recorder()
        timer = Timer()
        cache = LeaseCache(TTLCache(maxsize=10, ttl=2, timer=timer), cleanup=cleanup)
        cache["k"] = "v1"
        lease = cache.lease("k")
        timer.tick()
        timer.tick()
        timer.tick()  # item is now expired
        with self.assertRaises(KeyError):
            cache.lease("k")
        self.assertNotIn("k", cache)
        # existing lease remains usable until released
        self.assertEqual("v1", lease.value)
        self.assertEqual([], cleanup.calls)
        lease.release()
        self.assertEqual([("k", "v1")], cleanup.calls)

    def test_expire_method_cleans_unleased_items(self):
        cleanup = Recorder()
        timer = Timer()
        cache = LeaseCache(TTLCache(maxsize=10, ttl=1, timer=timer), cleanup=cleanup)
        cache["k"] = "v1"
        timer.tick()
        timer.tick()
        expired = cache.expire()
        self.assertEqual([("k", "v1")], list(expired))
        self.assertEqual([("k", "v1")], cleanup.calls)

    def test_double_release_is_noop(self):
        cleanup = Recorder()
        cache = LeaseCache(LRUCache(maxsize=10), cleanup=cleanup)
        cache["k"] = "v1"
        lease = cache.lease("k")
        del cache["k"]
        lease.release()
        lease.release()
        lease.release()
        self.assertEqual([("k", "v1")], cleanup.calls)

    def test_cleanup_at_most_once(self):
        cleanup = Recorder()
        cache = LeaseCache(LRUCache(maxsize=10), cleanup=cleanup)
        cache["k"] = "v1"
        lease = cache.lease("k")
        del cache["k"]
        cache["k"] = "v2"
        del cache["k"]  # v2: no leases, cleaned on delete
        lease.release()  # v1: cleaned on last release
        self.assertEqual([("k", "v1"), ("k", "v2")], sorted(cleanup.calls))
        # deleting again raises and cleans nothing more
        with self.assertRaises(KeyError):
            del cache["k"]
        self.assertEqual(2, len(cleanup.calls))

    def test_unleased_values_follow_underlying_eviction(self):
        cleanup = Recorder()
        cache = LeaseCache(LRUCache(maxsize=2), cleanup=cleanup)
        cache["a"] = 1
        cache["b"] = 2
        cache["c"] = 3  # evicts "a" per LRU rules
        self.assertEqual([("a", 1)], cleanup.calls)
        self.assertNotIn("a", cache)
        self.assertEqual(2, len(cache))

    def test_eviction_of_leased_value_defers_cleanup(self):
        cleanup = Recorder()
        cache = LeaseCache(LRUCache(maxsize=2), cleanup=cleanup)
        cache["a"] = 1
        lease = cache.lease("a")
        cache["b"] = 2
        cache["c"] = 3  # evicts "a", but it is still leased
        self.assertNotIn("a", cache)
        self.assertEqual([], cleanup.calls)
        self.assertEqual(1, lease.value)
        lease.release()
        self.assertEqual([("a", 1)], cleanup.calls)

    def test_clear_with_active_leases(self):
        cleanup = Recorder()
        cache = LeaseCache(LRUCache(maxsize=10), cleanup=cleanup)
        cache["a"] = 1
        cache["b"] = 2
        lease = cache.lease("b")
        cache.clear()
        self.assertEqual([("a", 1)], cleanup.calls)
        self.assertEqual(2, lease.value)
        lease.release()
        self.assertEqual([("a", 1), ("b", 2)], cleanup.calls)

    def test_popitem(self):
        cleanup = Recorder()
        cache = LeaseCache(LRUCache(maxsize=10), cleanup=cleanup)
        cache["a"] = 1
        key, value = cache.popitem()
        self.assertEqual(("a", 1), (key, value))
        self.assertEqual([("a", 1)], cleanup.calls)

    def test_no_cleanup_callback(self):
        cache = LeaseCache(LRUCache(maxsize=2))
        cache["a"] = 1
        lease = cache.lease("a")
        del cache["a"]
        lease.release()  # must not raise without a callback
        cache["b"] = 2
        cache["c"] = 3  # eviction without callback must not raise
        self.assertNotIn("a", cache)

    def test_cleanup_may_reenter_cache(self):
        # the cleanup callback runs outside the internal lock, so it
        # may access the cache again without deadlocking
        cache = LeaseCache(LRUCache(maxsize=10))
        seen = []

        def cleanup(key, value):
            seen.append((key, value))
            cache["reentered"] = value  # write back from the callback
            cache.lease("reentered").release()  # and lease from it

        cache._cleanup = cleanup
        cache["k"] = "v1"
        lease = cache.lease("k")
        del cache["k"]

        done = []
        # run the release in a thread with a timeout: a deadlock would
        # block the thread forever and fail the join assertion
        thread = threading.Thread(target=lambda: done.append(lease.release()))
        thread.start()
        thread.join(timeout=10)
        self.assertFalse(thread.is_alive())
        self.assertEqual([("k", "v1")], seen)
        self.assertEqual("v1", cache["reentered"])

    def test_concurrent_leases_with_barrier(self):
        # controllable barrier: both threads hold a lease before the
        # item is deleted; cleanup runs exactly once, after the last
        # release
        cleanup = Recorder()
        cache = LeaseCache(LRUCache(maxsize=10), cleanup=cleanup)
        cache["k"] = "v1"
        barrier = threading.Barrier(3)
        leases = []
        errors = []

        def worker():
            try:
                lease = cache.lease("k")
                leases.append(lease)
                barrier.wait(timeout=10)  # both leases acquired
                barrier.wait(timeout=10)  # item deleted by main thread
                lease.release()
            except Exception as exc:  # pragma: no cover  # noqa: BLE001
                errors.append(exc)

        threads = [threading.Thread(target=worker) for _ in range(2)]
        for t in threads:
            t.start()
        barrier.wait(timeout=10)  # wait until both leases are held
        self.assertEqual(2, len(leases))
        del cache["k"]
        self.assertEqual([], cleanup.calls)
        barrier.wait(timeout=10)  # let workers release
        for t in threads:
            t.join(timeout=10)
            self.assertFalse(t.is_alive())
        self.assertEqual([], errors)
        self.assertEqual([("k", "v1")], cleanup.calls)

    def test_concurrent_lease_and_delete_with_barrier(self):
        # a lease acquired before the delete stays valid; a lease
        # attempted after the delete fails with KeyError
        cleanup = Recorder()
        cache = LeaseCache(LRUCache(maxsize=10), cleanup=cleanup)
        cache["k"] = "v1"
        barrier = threading.Barrier(2)
        results = []

        def reader():
            lease = cache.lease("k")
            barrier.wait(timeout=10)  # main thread deletes now
            results.append(lease.value)
            lease.release()

        thread = threading.Thread(target=reader)
        thread.start()
        barrier.wait(timeout=10)
        del cache["k"]
        thread.join(timeout=10)
        self.assertFalse(thread.is_alive())
        self.assertEqual(["v1"], results)
        self.assertEqual([("k", "v1")], cleanup.calls)
        with self.assertRaises(KeyError):
            cache.lease("k")


if __name__ == "__main__":
    unittest.main()
