import threading
import unittest

from frame_output import FrameOutput


class FrameOutputTests(unittest.TestCase):
    def test_close_drains_all_frames_in_order(self):
        actual = []
        worker = FrameOutput(actual.append)
        for i in range(6):
            worker.submit(i)
        worker.close()
        self.assertEqual(actual, list(range(6)))
        self.assertEqual(worker.submitted, worker.completed)
        self.assertFalse(worker.thread.is_alive())

    def test_overflow_is_explicit_and_previous_frames_survive(self):
        entered, release = threading.Event(), threading.Event()
        actual = []
        def consume(value):
            entered.set()
            release.wait(3)
            actual.append(value)
        worker = FrameOutput(consume, capacity=1)
        try:
            worker.submit(0)
            self.assertTrue(entered.wait(2))
            worker.submit(1)
            with self.assertRaisesRegex(RuntimeError, 'backlog'):
                worker.submit(2)
        finally:
            release.set()
            worker.close()
        self.assertEqual(actual, [0, 1])

    def test_disk_error_is_propagated_and_close_joins(self):
        def fail(value):
            raise OSError('disk full')
        worker = FrameOutput(fail)
        worker.submit(0)
        with self.assertRaisesRegex(RuntimeError, 'incomplete'):
            worker.drain()
        with self.assertRaisesRegex(RuntimeError, 'incomplete'):
            worker.close()
        self.assertFalse(worker.thread.is_alive())


if __name__ == '__main__':
    unittest.main()
