class Chat:
    def __init__(self, file):
        self.file = file

    def _sanitize(self, s: str) -> str:
        # LCD can't display control chars like \n / \r / \t (they show up as weird symbols)
        return s.replace("\r", " ").replace("\n", " ").replace("\t", " ")

    def _shorten_prefix(self, line: str) -> str:
        # Optional: shorten prefixes so messages fit on a 20-char LCD
        if line.startswith("Non Impaired:"):
            return "NI: " + line[len("Non Impaired:"):].lstrip()
        if line.startswith("Blind:"):
            return "BL: " + line[len("Blind:"):].lstrip()
        if line.startswith("Deaf:"):
            return "DF: " + line[len("Deaf:"):].lstrip()
        return line

    def readMsgLCD(self, width: int = 20, height: int = 4) -> list:
        """
        Returns exactly `height` strings, each exactly `width` chars,
        ready to print to a 20x4 LCD.

        Format (bottom-up):
        - newest content ends up on the bottom row
        - older content is above it
        - no control chars (\n won't get sent to the LCD)
        - long lines wrap, and the *tail end* of the newest content stays closest to the bottom
        """
        budget = width * height

        self.file.seek(0)
        raw_lines = self.file.readlines()

        # Clean up file lines
        lines = []
        for ln in raw_lines:
            ln = ln.rstrip("\n").rstrip("\r")
            ln = self._sanitize(ln)
            ln = " ".join(ln.split())  # collapse weird spacing
            if ln == "":
                continue
            lines.append(self._shorten_prefix(ln))

        # Build chunks bottom-up so newest ends at bottom
        chunks = []
        used = 0

        for ln in reversed(lines):  # newest message first
            # Wrap into width chunks
            ln_chunks = [ln[i:i + width] for i in range(0, len(ln), width)] or [""]

            # Put the end of the message closest to the bottom
            for part in reversed(ln_chunks):
                if used >= budget:
                    break

                remaining = budget - used
                if len(part) <= remaining:
                    chunks.append(part)
                    used += len(part)
                else:
                    # only take what fits (tail end)
                    chunks.append(part[-remaining:])
                    used = budget

            if used >= budget:
                break

        # Convert to top->bottom display order
        display = list(reversed(chunks))

        # Keep only last `height` rows (defensive)
        if len(display) > height:
            display = display[-height:]

        # Pad TOP so chat sits at the bottom
        while len(display) < height:
            display.insert(0, "")

        # Right-pad so we overwrite old characters on LCD
        return [row.ljust(width)[:width] for row in display]

    def readMsg(self) -> list:
        """
        (Kept for compatibility) Returns lines as strings.
        Use readMsgLCD() for LCD output.
        """
        self.file.seek(0)
        lines = self.file.readlines()
        lines = [self._sanitize(ln) for ln in lines]
        return lines

    def writeMsg(self, msg: str, person: int):
        """
        Person int 1,2,3

        1 NonI
        2 Blind
        3 Deaf
        """
        msg = self._sanitize(msg).strip()

        prefix = ""
        if person == 1:
            prefix = "Non Impaired: "
        elif person == 2:
            prefix = "Blind: "
        elif person == 3:
            prefix = "Deaf: "

        # Always append newline so readlines() splits messages correctly
        self.file.seek(0, 2)  # go to end
        self.file.write(prefix + msg + "\n")
        self.file.flush()  # ensure it writes during execution
