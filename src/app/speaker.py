import subprocess
import threading
import time


class Speaker:
    def __init__(self, rate: int = 170, voice: str = "en-us"):
        self.rate = int(rate)
        self.voice = str(voice)

        self._lock = threading.Lock()
        self._pending = None
        self._stop = False
        self._speaking = threading.Event()

        self._t = threading.Thread(target=self._run, daemon=True)
        self._t.start()

    def playSound(self, text: str):
        text = str(text).strip()
        if text == "":
            return 0
        # keep only the latest request
        with self._lock:
            self._pending = text
        return 0

    def wait(self, timeout: float = 30.0):
        """Block until the speaker has finished the current utterance."""
        deadline = time.time() + timeout
        while time.time() < deadline:
            with self._lock:
                if self._pending is None and not self._speaking.is_set():
                    return True
            time.sleep(0.05)
        return False

    def close(self):
        self._stop = True
        with self._lock:
            self._pending = None
        try:
            self._t.join(timeout=1.0)
        except Exception:
            pass

    def _run(self):
        while not self._stop:
            text = None
            with self._lock:
                if self._pending is not None:
                    text = self._pending
                    self._pending = None

            if text is None:
                time.sleep(0.01)
                continue

            self._speaking.set()
            try:
                p1 = subprocess.Popen(
                    ["espeak-ng", "--stdout", "-s", str(self.rate), "-v", self.voice, text],
                    stdout=subprocess.PIPE,
                    stderr=subprocess.DEVNULL,
                )
                p2 = subprocess.Popen(
                    ["paplay"],
                    stdin=p1.stdout,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                )
                p1.stdout.close()
                p2.communicate()
            except Exception as e:
                print("Speaker error:", repr(e))
            finally:
                self._speaking.clear()
