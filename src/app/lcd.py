from time import sleep
import smbus2
from gpioExpander import gpioExpander

class ExpanderPin:
    def __init__(self, expander: gpioExpander, bank: str, pin: int):
        self.expander = expander
        self.bank = bank
        self.pin = pin

    def value(self, v: int):
        self.expander.set_pin(self.bank, self.pin, bool(v))

class LCD:
    def __init__(self, bus: smbus2.SMBus, expander_addr: int, bank: str):
        self.exp = gpioExpander(bus, expander_addr)

        # For your config (LCD1 on bank A, first expander)
        # RS = PA0, Enable = PA1, Data D4-D7 = PA2, PA3, PA4, PA5
        self.rs = ExpanderPin(self.exp, bank, 0)
        self.en = ExpanderPin(self.exp, bank, 1)
        self.data_pins = [ExpanderPin(self.exp, bank, pin) for pin in (2, 3, 4, 5)]

        self.init_lcd()

    def pulse_enable(self):
        self.en.value(1)
        sleep(0.001)
        self.en.value(0)
        sleep(0.001)

    def send_nibble(self, nibble: int):
        binstr = f"{nibble:04b}"
        bits = list(binstr[::-1])
        for i, pin in enumerate(self.data_pins):
            pin.value(int(bits[i]))
        self.pulse_enable()

    def send_byte(self, data: int, is_data=True):
        self.rs.value(1 if is_data else 0)
        high = data // 16
        low = data % 16
        self.send_nibble(high)
        self.send_nibble(low)

    def command(self, cmd: int):
        self.send_byte(cmd, is_data=False)

    def write_char(self, ch: str):
        self.send_byte(ord(ch), is_data=True)

    def clear(self):
        self.command(0x01)
        sleep(0.002)

    def move_to(self, row: int, col: int):
        row_offsets = [0x00, 0x40, 0x14, 0x54]
        self.command(0x80 | (col + row_offsets[row]))

    def putstr(self, s: str):
        for c in s:
            self.write_char(c)

    def init_lcd(self):
        sleep(0.05)
        self.send_nibble(0x03)
        sleep(0.005)
        self.send_nibble(0x03)
        sleep(0.005)
        self.send_nibble(0x03)
        self.send_nibble(0x02)

        self.command(0x28)
        self.command(0x08)
        self.command(0x01)
        sleep(0.002)
        self.command(0x06)
        self.command(0x0C)
