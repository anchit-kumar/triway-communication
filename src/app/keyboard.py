from collections import deque
from evdev import InputDevice, ecodes
import threading

class KeyboardReader:
    # --- US keyboard mapping: keycode -> unshifted char ---
    KEYCODE_TO_CHAR = {
        ecodes.KEY_A: 'a', ecodes.KEY_B: 'b', ecodes.KEY_C: 'c',
        ecodes.KEY_D: 'd', ecodes.KEY_E: 'e', ecodes.KEY_F: 'f',
        ecodes.KEY_G: 'g', ecodes.KEY_H: 'h', ecodes.KEY_I: 'i',
        ecodes.KEY_J: 'j', ecodes.KEY_K: 'k', ecodes.KEY_L: 'l',
        ecodes.KEY_M: 'm', ecodes.KEY_N: 'n', ecodes.KEY_O: 'o',
        ecodes.KEY_P: 'p', ecodes.KEY_Q: 'q', ecodes.KEY_R: 'r',
        ecodes.KEY_S: 's', ecodes.KEY_T: 't', ecodes.KEY_U: 'u',
        ecodes.KEY_V: 'v', ecodes.KEY_W: 'w', ecodes.KEY_X: 'x',
        ecodes.KEY_Y: 'y', ecodes.KEY_Z: 'z',

        ecodes.KEY_1: '1', ecodes.KEY_2: '2', ecodes.KEY_3: '3',
        ecodes.KEY_4: '4', ecodes.KEY_5: '5', ecodes.KEY_6: '6',
        ecodes.KEY_7: '7', ecodes.KEY_8: '8', ecodes.KEY_9: '9',
        ecodes.KEY_0: '0',

        ecodes.KEY_SPACE: ' ',
        ecodes.KEY_MINUS: '-',
        ecodes.KEY_EQUAL: '=',
        ecodes.KEY_LEFTBRACE: '[',
        ecodes.KEY_RIGHTBRACE: ']',
        ecodes.KEY_BACKSLASH: '\\',
        ecodes.KEY_SEMICOLON: ';',
        ecodes.KEY_APOSTROPHE: "'",
        ecodes.KEY_COMMA: ',',
        ecodes.KEY_DOT: '.',
        ecodes.KEY_SLASH: '/',
        ecodes.KEY_GRAVE: '`',

        # control keys as "text" tokens (optional)
        ecodes.KEY_ENTER: '\n',
        ecodes.KEY_TAB: '\t',
        ecodes.KEY_BACKSPACE: '\b',
    }

    # --- Shift transforms for non-letters on US layout ---
    SHIFT_MAP = {
        '1': '!', '2': '@', '3': '#', '4': '$', '5': '%',
        '6': '^', '7': '&', '8': '*', '9': '(', '0': ')',
        '-': '_', '=': '+',
        '[': '{', ']': '}',
        '\\': '|',
        ';': ':', "'": '"',
        ',': '<', '.': '>',
        '/': '?',
        '`': '~'
    }

    SHIFT_KEYS = {ecodes.KEY_LEFTSHIFT, ecodes.KEY_RIGHTSHIFT}

    def __init__(self, devicePath):
        """
        Main purpose of the class is to read keyboard presses without requiring
        a whole while True loop based on it.

        We do this with a thread that runs along with our main code that
        references the function which captures inputs.
        """
        self.dev = InputDevice(devicePath)
        self._lock = threading.Lock()
        self._queue = deque()
        self._shift_down = False

        self.thread = threading.Thread(target=self.reader, daemon=True)
        self.thread.start()

    def _event_to_char(self, e, shift_down):
        """
        Convert a *key-down* event to a printable character (US layout).
        Returns None if not a text key or not a key-down event.
        """
        if e.value != 1:  # only key DOWN
            return None

        base = self.KEYCODE_TO_CHAR.get(e.code)
        if base is None:
            return None

        if shift_down:
            if base.isalpha():
                return base.upper()
            return self.SHIFT_MAP.get(base, base)

        return base

    def reader(self):
        try:
            for event in self.dev.read_loop():
                if event.type != ecodes.EV_KEY:
                    continue

                with self._lock:
                    if event.code in self.SHIFT_KEYS:
                        self._shift_down = (event.value != 0)

                    self._queue.append((event, self._shift_down))
        except Exception as e:
            print(f"[KeyboardReader] reader thread died for {self.dev.path}: {e}")


    def getKeyInfo(self):
        """
        Dict with key and state (hold, up, down) + optional 'char'
        """
        with self._lock:
            if not self._queue:
                return None

            e, shift_at_press = self._queue.popleft()
            key_name = ecodes.KEY.get(e.code, str(e.code))
            if isinstance(key_name, list):
                key_name = key_name[-1]  # normalize weird cases like KEY_MUTE

            return {
                "code": e.code,
                "name": key_name,
                "state": {0: "up", 1: "down", 2: "hold"}.get(e.value, f"value={e.value}"),
                "char": self._event_to_char(e, shift_at_press),
            }
