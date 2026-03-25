from time import sleep
import time
import os
from pathlib import Path

import smbus2
from evdev import InputDevice, list_devices

from lcd import LCD
from speaker import Speaker
from keyboard import KeyboardReader
from chat import Chat
from camera import ASLCamera


# ----------------------------
# Debug
# ----------------------------
DEBUG = True

def dprint(*args):
    if DEBUG:
        print("[DBG]", *args)


# ----------------------------
# Config
# ----------------------------
LCD_W = 20
LCD_H = 4

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CHAT_PATH = os.path.join(BASE_DIR, "chat.txt")

# Keyboard name fragments (from `evtest` / `lsinput`)
NONI_KBD_NAME = "LiteOn Lenovo New Calliope USB Keyboard"
BLIND_KBD_NAME = "LITE-ON Technology USB NetVista Full Width Keyboard"

# Camera config
_MODELS = Path(__file__).parent.parent / "models"
CAMERA_CFG = dict(
    keypoints_model_path=str(_MODELS / "hand_landmarker.task"),
    asl_model_path=str(_MODELS / "mlp_best.keras"),
    labels_path=str(_MODELS / "labels.json"),
    device="/dev/video0",
    width=1280,
    height=720,
    fps=30,
    print_every=0,
)

# Deaf ASL typing settings
ASL_CONF_MIN = 0.80
ASL_WINDOW = 12
ASL_NEED = 10
ASL_NOHAND_SECONDS = 0.7
ASL_SEND_SECONDS = 5  # send after 5 seconds continuous no-hand (if text exists)

# Blind message detection polling (backup to os.stat)
CHAT_POLL_SECONDS = 0.25


# ----------------------------
# LCD helper
# ----------------------------
def writeToLCD(content, mode):
    """
    Writes either:
      - a string (typing mode): wraps last 80 chars onto 20x4 (top-down)
      - a list of up to 4 strings (chatroom mode): bottom-anchored
    Always pads to 20 chars so old characters don't stay on screen.
    """
    if isinstance(content, list):
        rows = content[:]
        while len(rows) < LCD_H:
            rows.insert(0, "")
        rows = rows[-LCD_H:]
    else:
        s = str(content)
        s = s.replace("\r", " ").replace("\n", " ").replace("\t", " ")

        if len(s) > LCD_W * LCD_H:
            s = s[-(LCD_W * LCD_H):]

        rows = [s[i:i + LCD_W] for i in range(0, len(s), LCD_W)]
        while len(rows) < LCD_H:
            rows.append("")
        rows = rows[:LCD_H]

    if mode == "NonI":
        for i in range(LCD_H):
            lcdNonI.move_to(i, 0)
            lcdNonI.putstr(rows[i].ljust(LCD_W)[:LCD_W])

    if mode == "Deaf":
        for i in range(LCD_H):
            lcdDeaf.move_to(i, 0)
            lcdDeaf.putstr(rows[i].ljust(LCD_W)[:LCD_W])


# ----------------------------
# Device discovery helpers
# ----------------------------
def find_keyboard_path(name_fragment: str):
    fallback = ""

    for path in list_devices():
        dev = InputDevice(path)
        name = dev.name or ""

        if "Consumer Control" in name:
            continue

        if name == name_fragment:
            return path

        if (fallback == "") and (name_fragment in name):
            fallback = path

    return fallback


def ensure_chat_file(path: str):
    if not os.path.exists(path):
        with open(path, "w") as f:
            f.write("")


# ----------------------------
# Init hardware + IO
# ----------------------------
ensure_chat_file(CHAT_PATH)
dprint("CHAT_PATH =", CHAT_PATH)

nonI_path = find_keyboard_path(NONI_KBD_NAME)
blind_path = find_keyboard_path(BLIND_KBD_NAME)

if nonI_path == "" or blind_path == "":
    print("ERROR: Could not find one or both keyboards.")
    print(f"  NonI wanted:  {NONI_KBD_NAME!r}  -> found path={nonI_path!r}")
    print(f"  Blind wanted: {BLIND_KBD_NAME!r} -> found path={blind_path!r}")
    print("\nDetected input devices:")
    for p in list_devices():
        try:
            d = InputDevice(p)
            print(f"  - {p}: {d.name}")
        except Exception:
            print(f"  - {p}: <unable to open>")
    raise SystemExit(1)

dprint("nonI_path =", nonI_path)
dprint("blind_path =", blind_path)

listenerNonI = KeyboardReader(nonI_path)
listenerBlind = KeyboardReader(blind_path)

bus = smbus2.SMBus(7)
lcdNonI = LCD(bus, 0x23, 'B')
lcdDeaf = LCD(bus, 0x21, 'A')

chat_file = open(CHAT_PATH, "r+")
chat = Chat(chat_file)

cam = ASLCamera(**CAMERA_CFG)
audio = Speaker()
audio.playSound("Hello!")


# ----------------------------
# State (Non-I)
# ----------------------------
str1 = ""
if1Idle = True
forceLCD = True

# ----------------------------
# State (Blind) — clean pipeline
# ----------------------------
blindText = ""
blindWord = ""

blind_state = "IDLE"        # IDLE / TYPING / PROMPT
blind_prev_state = "IDLE"

msgCache = chat.readMsg()
blind_ack_len = len(msgCache)   # messages acknowledged (after Y or N)
blind_prompt_for_len = blind_ack_len  # length at time we prompted (debug/consistency)

dprint("startup chat lines =", blind_ack_len)

# ----------------------------
# State (Chat detection)
# ----------------------------
lastChatSize = -1
lastChatMTime = -1

try:
    st0 = os.stat(CHAT_PATH)
    lastChatSize = st0.st_size
    lastChatMTime = st0.st_mtime_ns
except Exception:
    pass

next_poll = time.time() + CHAT_POLL_SECONDS


# ----------------------------
# State (Deaf / ASL)
# ----------------------------
deafText = ""
deafIdle = True
forceLCDDeaf = True

asl_state = "WAIT_CHAR"
asl_win = []
asl_nohand_start = None
asl_nohand_start_send = None

print("starting")


# ----------------------------
# Helper: trigger blind prompt
# ----------------------------
def trigger_blind_prompt(num_new: int, cur_len: int):
    global blind_state, blind_prev_state, blind_prompt_for_len
    if blind_state != "PROMPT":
        blind_prev_state = blind_state
    blind_state = "PROMPT"
    blind_prompt_for_len = cur_len
    audio.playSound(f"You have {num_new} new messages. Press Y to hear or N to ignore.")
    dprint("PROMPT:", num_new, "new messages; state was", blind_prev_state)


# ----------------------------
# Main loop
# ----------------------------
try:
    while True:
        didWork = False

        # ---- Detect chat changes using os.stat() ----
        chatChanged = False
        try:
            st = os.stat(CHAT_PATH)
            if (st.st_size != lastChatSize) or (st.st_mtime_ns != lastChatMTime):
                lastChatSize = st.st_size
                lastChatMTime = st.st_mtime_ns
                chatChanged = True
        except Exception:
            pass

        if chatChanged:
            msgCache = chat.readMsg()
            dprint("chatChanged(stat): len=", len(msgCache), "ack=", blind_ack_len, "state=", blind_state)

            if len(msgCache) > blind_ack_len:
                trigger_blind_prompt(len(msgCache) - blind_ack_len, len(msgCache))

            if if1Idle:
                forceLCD = True
            if deafIdle:
                forceLCDDeaf = True

        # ---- Backup poll (catches missed stat updates) ----
        now = time.time()
        if now >= next_poll:
            next_poll = now + CHAT_POLL_SECONDS
            try:
                cur = chat.readMsg()
                if len(cur) != len(msgCache):
                    msgCache = cur
                    dprint("chatChanged(poll): len=", len(msgCache), "ack=", blind_ack_len, "state=", blind_state)

                if len(msgCache) > blind_ack_len and blind_state != "PROMPT":
                    trigger_blind_prompt(len(msgCache) - blind_ack_len, len(msgCache))
            except Exception as e:
                dprint("poll error:", repr(e))

        # ---- Idle LCD refresh for Non-I ----
        if if1Idle and forceLCD:
            rows = chat.readMsgLCD(LCD_W, LCD_H)
            writeToLCD(rows, "NonI")
            forceLCD = False

        # ---- Idle LCD refresh for Deaf ----
        if deafIdle and forceLCDDeaf:
            rowsD = chat.readMsgLCD(LCD_W, LCD_H)
            writeToLCD(rowsD, "Deaf")
            forceLCDDeaf = False

        # ---- Read keyboard events ----
        evt1 = listenerNonI.getKeyInfo()
        evt2 = listenerBlind.getKeyInfo()

        # ==========================
        # Non-I loop (typing + send)
        # ==========================
        e1 = evt1
        while e1:
            if (e1["state"] == "down" or e1["state"] == "hold"):
                didWork = True
                if1Idle = False

                if str1 == "":
                    lcdNonI.clear()

                if e1["char"]:
                    if e1["code"] == 14:
                        str1 = str1[:-1]
                    elif e1["code"] == 28:
                        if str1.strip():
                            chat.writeMsg(str1, 1)
                        str1 = ""
                        if1Idle = True
                        forceLCD = True
                        if deafIdle:
                            forceLCDDeaf = True

                        # refresh caches
                        msgCache = chat.readMsg()
                        try:
                            st2 = os.stat(CHAT_PATH)
                            lastChatSize = st2.st_size
                            lastChatMTime = st2.st_mtime_ns
                        except Exception:
                            pass
                    else:
                        str1 += e1["char"]

                    writeToLCD(str1, "NonI")

            e1 = listenerNonI.getKeyInfo()

        # ==========================
        # Blind loop (IDLE / TYPING / PROMPT)
        # ==========================
        e2 = evt2
        while e2:
            if e2["state"] != "down":
                e2 = listenerBlind.getKeyInfo()
                continue

            didWork = True

            # PROMPT: only accept Y/N
            if blind_state == "PROMPT":
                dprint("PROMPT key:", e2.get("name"), e2.get("code"))

                if e2["code"] == 21:  # Y
                    cur = chat.readMsg()
                    msgCache = cur
                    pending = cur[blind_ack_len:]

                    dprint("Y pressed; speaking", len(pending), "lines")

                    for line in pending:
                        speakLine = line.replace("\n", " ").replace("\r", " ").strip()
                        if speakLine:
                            audio.playSound(speakLine)
                            audio.wait()

                    blind_ack_len = len(cur)
                    blind_state = blind_prev_state if blind_prev_state in ("IDLE", "TYPING") else "IDLE"
                    dprint("PROMPT done ->", blind_state, "ack_len=", blind_ack_len)

                elif e2["code"] == 49:  # N
                    cur = chat.readMsg()
                    msgCache = cur
                    blind_ack_len = len(cur)
                    blind_state = blind_prev_state if blind_prev_state in ("IDLE", "TYPING") else "IDLE"
                    dprint("N pressed ->", blind_state, "ack_len=", blind_ack_len)

                e2 = listenerBlind.getKeyInfo()
                continue

            # IDLE -> any meaningful key starts typing
            if blind_state == "IDLE":
                if (e2.get("char")) or (e2["code"] in (14, 57, 28)):
                    blind_state = "TYPING"
                    dprint("blind -> TYPING")

            if blind_state == "TYPING":
                if e2["code"] == 14:  # Backspace
                    if blindText:
                        lastChar = blindText[-1]
                        blindText = blindText[:-1]

                        if lastChar == " ":
                            blindWord = blindText.rsplit(" ", 1)[-1] if " " in blindText else blindText
                        else:
                            blindWord = blindWord[:-1] if blindWord else ""

                elif e2["code"] == 28:  # Enter (send)
                    if blindText.strip():
                        chat.writeMsg(blindText, 2)

                    blindText = ""
                    blindWord = ""
                    blind_state = "IDLE"
                    dprint("blind sent -> IDLE")

                    # prevent self-notify
                    msgCache = chat.readMsg()
                    blind_ack_len = len(msgCache)

                    try:
                        st2 = os.stat(CHAT_PATH)
                        lastChatSize = st2.st_size
                        lastChatMTime = st2.st_mtime_ns
                    except Exception:
                        pass

                elif e2["code"] == 57:  # Space: speak previous word
                    if blindWord:
                        audio.playSound(blindWord)
                    if (not blindText) or blindText[-1] != " ":
                        blindText += " "
                    blindWord = ""

                else:
                    if e2.get("char"):
                        blindText += e2["char"]
                        blindWord += e2["char"]

            e2 = listenerBlind.getKeyInfo()

        # ==========================
        # Deaf ASL loop (commit + send)
        # ==========================
        pred = cam.predict_cur_letter()

        if pred is None:
            asl_label, asl_prob = None, 0.0
        else:
            asl_label, asl_prob = pred[0], float(pred[1])

        if asl_label is None or asl_prob < ASL_CONF_MIN:
            asl_label = None

        # send timer
        if deafText.strip() != "":
            if asl_label is None:
                if asl_nohand_start_send is None:
                    asl_nohand_start_send = time.time()
                elif (time.time() - asl_nohand_start_send) >= ASL_SEND_SECONDS:
                    chat.writeMsg(deafText, 3)
                    deafText = ""
                    msgCache = chat.readMsg()

                    try:
                        st2 = os.stat(CHAT_PATH)
                        lastChatSize = st2.st_size
                        lastChatMTime = st2.st_mtime_ns
                    except Exception:
                        pass

                    deafIdle = True
                    forceLCDDeaf = True

                    asl_state = "WAIT_CHAR"
                    asl_win = []
                    asl_nohand_start = None
                    asl_nohand_start_send = None
            else:
                asl_nohand_start_send = None
        else:
            asl_nohand_start_send = None

        # commit logic
        if asl_state == "WAIT_CHAR":
            asl_win.append(asl_label)
            if len(asl_win) > ASL_WINDOW:
                asl_win.pop(0)

            mode = None
            mode_count = 0
            for x in asl_win:
                if x is None:
                    continue
                c = 0
                for y in asl_win:
                    if y == x:
                        c += 1
                if c > mode_count:
                    mode_count = c
                    mode = x

            if mode is not None and mode_count >= ASL_NEED:
                didWork = True
                if deafIdle:
                    lcdDeaf.clear()
                    deafIdle = False

                m = str(mode).lower()
                if m == "del":
                    deafText = deafText[:-1]
                elif m == "space":
                    deafText += " "
                else:
                    deafText += mode

                writeToLCD(deafText, "Deaf")

                asl_state = "WAIT_NOHAND"
                asl_win = []
                asl_nohand_start = None

        else:  # WAIT_NOHAND
            if asl_label is None:
                nowt = time.time()
                if asl_nohand_start is None:
                    asl_nohand_start = nowt

                if (nowt - asl_nohand_start) >= ASL_NOHAND_SECONDS:
                    asl_state = "WAIT_CHAR"
                    asl_win = []
                    asl_nohand_start = None
                    if not deafText:
                        deafIdle = True
                        forceLCDDeaf = True
            else:
                asl_nohand_start = None

        # ---- Sleep ----
        if didWork:
            time.sleep(0.001)
        else:
            time.sleep(0.01)

finally:
    try:
        cam.close()
    except Exception:
        pass
    try:
        chat_file.close()
    except Exception:
        pass
