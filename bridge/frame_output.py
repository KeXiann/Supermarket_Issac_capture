"""Ordered bounded output worker; no renderer or simulator calls leave the main thread."""
import queue
import threading


class FrameOutput:
    def __init__(self, consume, capacity=8):
        self.consume = consume
        self.queue = queue.Queue(maxsize=capacity)
        self.error = None
        self.closed = False
        self.submitted = self.completed = 0
        self.thread = threading.Thread(target=self._run, name='ego-frame-output', daemon=True)
        self.thread.start()

    def _run(self):
        while True:
            item = self.queue.get()
            try:
                if item is None:
                    return
                if self.error is None:
                    self.consume(item)
                    self.completed += 1
            except BaseException as exc:
                self.error = exc
            finally:
                self.queue.task_done()

    def check(self):
        if self.error is not None:
            raise RuntimeError('ego frame output failed; recording is incomplete') from self.error

    def submit(self, item):
        self.check()
        if self.closed:
            raise RuntimeError('ego frame output already closed')
        try:
            self.queue.put_nowait(item)
        except queue.Full as exc:
            raise RuntimeError('ego output backlog exceeded 8 frames; refusing silent video loss') from exc
        self.submitted += 1

    def drain(self):
        self.queue.join()
        self.check()

    def close(self):
        if not self.closed:
            self.queue.join()
            self.closed = True
            self.queue.put(None)
            self.thread.join()
        self.check()
