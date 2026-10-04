import threading
import unittest

import cachetools


class Cleanup:
    def __init__(self):
        self.calls = []

    def __call__(self, key, value):
        self.calls.append((key, value))

    def count(self, key=None):
        if key is None:
            return len(self.calls)
        return sum(1 for k, _ in self.calls if k == key)


class LeasedCacheTest(unittest.TestCase):
    def test_lease_returns_value(self):
        cleanup = Cleanup()
        cache = cachetools.LeasedCache(cachetools.LRUCache(maxsize=2), cleanup)
        cache[1] = "v1"

        lease = cache.lease(1)
        self.assertEqual("v1", lease.value)
        self.assertEqual(1, lease.key)
        self.assertEqual(1, lease.generation)
        self.assertFalse(lease.released)

        lease.release()
        self.assertTrue(lease.released)
        # the entry is still cached, so no cleanup yet
        self.assertEqual(0, cleanup.count())

    def test_lease_cleaned_up_after_delete_and_release(self):
        cleanup = Cleanup()
        cache = cachetools.LeasedCache(cachetools.LRUCache(maxsize=2), cleanup)
        cache[1] = "v1"

        lease = cache.lease(1)
        del cache[1]
        lease.release()
        self.assertEqual([(1, "v1")], cleanup.calls)

    def test_lease_unpacks_to_value_and_lease(self):
        cache = cachetools.LeasedCache(cachetools.LRUCache(maxsize=2))
        cache[1] = "v1"

        value, lease = cache.lease(1)
        self.assertEqual("v1", value)
        self.assertEqual(1, lease.key)
        lease.release()

    def test_delete_defers_cleanup_until_release(self):
        cleanup = Cleanup()
        cache = cachetools.LeasedCache(cachetools.LRUCache(maxsize=2), cleanup)
        cache[1] = "v1"

        lease = cache.lease(1)
        del cache[1]
        # no cleanup while lease is outstanding
        self.assertEqual(0, cleanup.count())
        # value still usable through the lease
        self.assertEqual("v1", lease.value)

        cache[1] = "v2"
        # releasing the old lease only cleans up v1
        lease.release()
        self.assertEqual([(1, "v1")], cleanup.calls)
        # v2 is still readable
        self.assertEqual("v2", cache[1])

    def test_update_creates_new_generation(self):
        cleanup = Cleanup()
        cache = cachetools.LeasedCache(cachetools.LRUCache(maxsize=2), cleanup)
        cache[1] = "v1"

        lease = cache.lease(1)
        cache[1] = "v2"
        self.assertEqual(1, lease.generation)  # lease keeps old generation
        self.assertEqual(0, cleanup.count())

        lease.release()
        self.assertEqual([(1, "v1")], cleanup.calls)
        self.assertEqual("v2", cache[1])

    def test_two_leases_wait_for_last_release(self):
        cleanup = Cleanup()
        cache = cachetools.LeasedCache(cachetools.LRUCache(maxsize=2), cleanup)
        cache[1] = "v1"

        lease1 = cache.lease(1)
        lease2 = cache.lease(1)
        self.assertEqual(1, lease1.generation)
        self.assertEqual(1, lease2.generation)

        del cache[1]
        lease1.release()
        self.assertEqual(0, cleanup.count())
        lease2.release()
        self.assertEqual(1, cleanup.count())
        self.assertEqual([(1, "v1")], cleanup.calls)

    def test_cleanup_at_most_once(self):
        cleanup = Cleanup()
        cache = cachetools.LeasedCache(cachetools.LRUCache(maxsize=2), cleanup)
        cache[1] = "v1"

        lease = cache.lease(1)
        del cache[1]
        self.assertTrue(lease.release())
        # repeated release must not trigger cleanup again
        self.assertFalse(lease.release())
        self.assertFalse(lease.release())
        self.assertEqual(1, cleanup.count())

    def test_no_new_lease_after_delete(self):
        cache = cachetools.LeasedCache(cachetools.LRUCache(maxsize=2))
        cache[1] = "v1"

        lease = cache.lease(1)
        del cache[1]
        with self.assertRaises(KeyError):
            cache.lease(1)
        # the existing lease remains usable
        self.assertEqual("v1", lease.value)
        lease.release()

    def test_no_new_lease_after_expiry(self):
        now = [0.0]
        cleanup = Cleanup()
        inner = cachetools.TTLCache(maxsize=10, ttl=10, timer=lambda: now[0])
        cache = cachetools.LeasedCache(inner, cleanup)
        cache[1] = "v1"

        lease = cache.lease(1)
        now[0] = 11  # expire the entry
        with self.assertRaises(KeyError):
            cache.lease(1)
        # the existing lease remains usable until released
        self.assertEqual("v1", lease.value)
        lease.release()
        self.assertEqual(1, cleanup.count())

    def test_expiry_without_lease_no_cleanup(self):
        now = [0.0]
        cleanup = Cleanup()
        inner = cachetools.TTLCache(maxsize=10, ttl=10, timer=lambda: now[0])
        cache = cachetools.LeasedCache(inner, cleanup)
        cache[1] = "v1"

        now[0] = 11  # expire the entry
        with self.assertRaises(KeyError):
            cache.lease(1)
        self.assertEqual(0, cleanup.count())

    def test_plain_values_follow_eviction_rules(self):
        cleanup = Cleanup()
        cache = cachetools.LeasedCache(cachetools.LRUCache(maxsize=2), cleanup)
        cache[1] = "v1"
        cache[2] = "v2"
        cache[3] = "v3"  # evicts key 1 per LRU rules

        self.assertNotIn(1, cache)
        self.assertEqual(0, cleanup.count())
        self.assertEqual("v2", cache[2])
        self.assertEqual("v3", cache[3])

    def test_eviction_with_lease_defers_cleanup(self):
        cleanup = Cleanup()
        cache = cachetools.LeasedCache(cachetools.LRUCache(maxsize=2), cleanup)
        cache[1] = "v1"
        lease = cache.lease(1)
        cache[2] = "v2"
        cache[3] = "v3"  # evicts key 1, but lease is outstanding

        self.assertNotIn(1, cache)
        self.assertEqual(0, cleanup.count())
        self.assertEqual("v1", lease.value)
        lease.release()
        self.assertEqual(1, cleanup.count())
        self.assertEqual([(1, "v1")], cleanup.calls)

    def test_clear_with_leases(self):
        cleanup = Cleanup()
        cache = cachetools.LeasedCache(cachetools.LRUCache(maxsize=2), cleanup)
        cache[1] = "v1"
        lease = cache.lease(1)

        cache.clear()
        self.assertEqual(0, cleanup.count())
        self.assertEqual("v1", lease.value)
        lease.release()
        self.assertEqual(1, cleanup.count())

    def test_pop_with_lease(self):
        cleanup = Cleanup()
        cache = cachetools.LeasedCache(cachetools.LRUCache(maxsize=2), cleanup)
        cache[1] = "v1"
        lease = cache.lease(1)

        self.assertEqual("v1", cache.pop(1))
        self.assertEqual(0, cleanup.count())
        lease.release()
        self.assertEqual(1, cleanup.count())

    def test_context_manager(self):
        cleanup = Cleanup()
        cache = cachetools.LeasedCache(cachetools.LRUCache(maxsize=2), cleanup)
        cache[1] = "v1"

        with cache.lease(1) as lease:
            self.assertEqual("v1", lease.value)
            del cache[1]
        self.assertTrue(lease.released)
        self.assertEqual(1, cleanup.count())

    def test_context_manager_still_cached_no_cleanup(self):
        cleanup = Cleanup()
        cache = cachetools.LeasedCache(cachetools.LRUCache(maxsize=2), cleanup)
        cache[1] = "v1"

        with cache.lease(1) as lease:
            self.assertEqual("v1", lease.value)
        self.assertTrue(lease.released)
        self.assertEqual(0, cleanup.count())

    def test_context_manager_unpack(self):
        cache = cachetools.LeasedCache(cachetools.LRUCache(maxsize=2))
        cache[1] = "v1"

        with cache.lease(1) as (value, lease):
            self.assertEqual("v1", value)
        self.assertTrue(lease.released)

    def test_cleanup_callback_reentrant(self):
        # The cleanup callback runs outside the internal lock, so it
        # may safely re-enter the cache without deadlocking.
        def cleanup(key, value):
            self.assertEqual("v1", value)
            # re-enter the cache from within the callback
            self.assertEqual(1, len(cache))
            self.assertEqual("v2", cache[2])
            lease = cache.lease(2)
            lease.release()

        cache = cachetools.LeasedCache(cachetools.LRUCache(maxsize=2), cleanup)
        cache[1] = "v1"
        cache[2] = "v2"
        lease = cache.lease(1)
        del cache[1]
        lease.release()  # must not deadlock

    def test_cleanup_callback_reentrant_writes(self):
        def cleanup(key, value):
            cache[3] = "v3"
            self.assertEqual("v3", cache[3])

        cache = cachetools.LeasedCache(cachetools.LRUCache(maxsize=3), cleanup)
        cache[1] = "v1"
        lease = cache.lease(1)
        del cache[1]
        lease.release()  # must not deadlock
        self.assertEqual("v3", cache[3])

    def test_generation_lifecycle(self):
        cleanup = Cleanup()
        cache = cachetools.LeasedCache(cachetools.LRUCache(maxsize=4), cleanup)

        # generation 1: leased, then replaced by generation 2
        cache[1] = "v1"
        lease1 = cache.lease(1)
        cache[1] = "v2"
        # generation 2: leased, then deleted
        lease2 = cache.lease(1)
        del cache[1]
        # generation 3: new value, no lease
        cache[1] = "v3"

        self.assertEqual(0, cleanup.count())
        # releasing lease1 cleans up v1 only
        lease1.release()
        self.assertEqual([(1, "v1")], cleanup.calls)
        # v2 still held by lease2, v3 readable
        self.assertEqual("v3", cache[1])
        # releasing lease2 cleans up v2
        lease2.release()
        self.assertEqual(2, cleanup.count())
        self.assertEqual([(1, "v1"), (1, "v2")], cleanup.calls)
        self.assertEqual("v3", cache[1])

    def test_multiple_keys_independent(self):
        cleanup = Cleanup()
        cache = cachetools.LeasedCache(cachetools.LRUCache(maxsize=4), cleanup)
        cache[1] = "v1"
        cache[2] = "v2"

        lease1 = cache.lease(1)
        lease2 = cache.lease(2)
        del cache[1]

        lease2.release()  # no cleanup for key 2 (still in cache)
        self.assertEqual(0, cleanup.count())
        lease1.release()  # cleanup key 1
        self.assertEqual(1, cleanup.count(1))
        self.assertEqual(0, cleanup.count(2))

    def test_thread_barrier(self):
        cleanup = Cleanup()
        cache = cachetools.LeasedCache(cachetools.LRUCache(maxsize=2), cleanup)
        cache[1] = "v1"

        # 3 parties: 2 worker threads + the main thread
        leases_held = threading.Barrier(3)
        release_done = threading.Barrier(3)
        may_release = threading.Event()

        def worker():
            lease = cache.lease(1)
            leases_held.wait()  # both leases are now held
            may_release.wait()  # wait for the main thread's signal
            lease.release()
            release_done.wait()  # both leases have been released

        threads = [threading.Thread(target=worker) for _ in range(2)]
        for t in threads:
            t.start()

        leases_held.wait()
        # both leases held: cleanup must not have run
        self.assertEqual(0, cleanup.count())

        # delete the entry while both leases are outstanding; cleanup
        # must be deferred until the last lease is released
        del cache[1]
        may_release.set()
        release_done.wait()
        for t in threads:
            t.join()
        # both leases released: cleanup must have run exactly once
        self.assertEqual(1, cleanup.count())
        self.assertEqual([(1, "v1")], cleanup.calls)

    def test_generations_cleaned_independently(self):
        cleanup = Cleanup()
        cache = cachetools.LeasedCache(cachetools.LRUCache(maxsize=2), cleanup)
        cache[1] = "v1"

        lease1 = cache.lease(1)  # generation 1
        cache[1] = "v2"  # generation 2; generation 1 is retired
        lease2 = cache.lease(1)  # generation 2
        del cache[1]  # both generations are now dead

        lease1.release()  # cleanup generation 1 only
        self.assertEqual([(1, "v1")], cleanup.calls)
        lease2.release()  # cleanup generation 2 as well
        self.assertEqual([(1, "v1"), (1, "v2")], cleanup.calls)


if __name__ == "__main__":
    unittest.main()
