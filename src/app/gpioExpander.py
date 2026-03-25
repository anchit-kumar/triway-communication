import smbus2

class gpioExpander:
    _REG_IODIRA = 0x00
    _REG_IODIRB = 0x01
    _REG_OLATA  = 0x14
    _REG_OLATB  = 0x15
    _REG_GPIOA  = 0x12
    _REG_GPIOB  = 0x13

    def __init__(self, bus: smbus2.SMBus, address: int):
        self.bus = bus
        self.address = address
        # configure all pins of both ports as outputs
        self.bus.write_byte_data(self.address, self._REG_IODIRA, 0x00)
        self.bus.write_byte_data(self.address, self._REG_IODIRB, 0x00)
        # internal state strings for ports A and B
        self._state_a = "00000000"
        self._state_b = "00000000"
        # sync expander to initial state (all low)
        self.bus.write_byte_data(self.address, self._REG_OLATA, 0x00)
        self.bus.write_byte_data(self.address, self._REG_OLATB, 0x00)

    def _write_state(self, bank: str):
        """Write the current internal state string to the expander."""
        if bank == 'A':
            byte = int(self._state_a, 2)
            self.bus.write_byte_data(self.address, self._REG_OLATA, byte)
        elif bank == 'B':
            byte = int(self._state_b, 2)
            self.bus.write_byte_data(self.address, self._REG_OLATB, byte)
        else:
            raise ValueError("Bank must be 'A' or 'B'")

    def set_pin(self, bank: str, pin: int, high: bool):
        """
        Set or clear a single pin on bank 'A' or 'B'.
        pin: 0–7  (0 = least-significant bit, 7 = most-significant bit)
        """
        if pin < 0 or pin > 7:
            raise ValueError("Pin index must be 0–7")
        if bank == 'A':
            s = list(self._state_a)
            idx = 7 - pin  # map pin 0 → rightmost, pin 7 → leftmost
            s[idx] = '1' if high else '0'
            self._state_a = "".join(s)
            self._write_state('A')
        elif bank == 'B':
            s = list(self._state_b)
            idx = 7 - pin
            s[idx] = '1' if high else '0'
            self._state_b = "".join(s)
            self._write_state('B')
        else:
            raise ValueError("Bank must be 'A' or 'B'")

    def read_bank_str(self, bank: str) -> str:
        """Return an 8-bit string representing the state of the bank (MSB first)."""
        if bank == 'A':
            byte = self.bus.read_byte_data(self.address, self._REG_GPIOA)
        elif bank == 'B':
            byte = self.bus.read_byte_data(self.address, self._REG_GPIOB)
        else:
            raise ValueError("Bank must be 'A' or 'B'")
        return format(byte, '08b')
