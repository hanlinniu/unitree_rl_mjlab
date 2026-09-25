#!/usr/bin/env python3
"""Live joystick left/right stick monitor.

Reads /dev/input/js* (Linux) and prints stick commands in the terminal.
Default axis map matches common Xbox-style pads on Linux (incl. many
PC/PS3 Android clones):

  Left stick  : axis 0 (X), axis 1 (Y)
  Right stick : axis 3 (X), axis 4 (Y)
  Triggers    : axis 2 / 5  (often rest at -1.0)

Examples:
  python scripts/joystick_stick_monitor.py
  python scripts/joystick_stick_monitor.py --device /dev/input/js0
  python scripts/joystick_stick_monitor.py --lx 0 --ly 1 --rx 2 --ry 3
"""

from __future__ import annotations

import argparse
import os
import select
import struct
import sys
import time
from pathlib import Path

# Linux joystick event: uint32 time, int16 value, uint8 type, uint8 number
_JS_EVENT = struct.Struct("IhBB")
_JS_EVENT_AXIS = 0x02
_JS_EVENT_BUTTON = 0x01
_JS_EVENT_INIT = 0x80


def _find_device(explicit: str | None) -> str:
  if explicit:
    if not Path(explicit).exists():
      raise FileNotFoundError(f"Joystick device not found: {explicit}")
    return explicit
  for p in sorted(Path("/dev/input").glob("js*")):
    return str(p)
  raise FileNotFoundError(
    "No /dev/input/js* found. Plug in the gamepad and check `ls /dev/input/js*`."
  )


def _bar(v: float, width: int = 21) -> str:
  """ASCII bar for value in [-1, 1]."""
  v = max(-1.0, min(1.0, v))
  mid = width // 2
  pos = int(round((v + 1.0) * 0.5 * (width - 1)))
  chars = ["·"] * width
  chars[mid] = "|"
  chars[pos] = "█"
  return "".join(chars)


def _deadzone(v: float, dz: float) -> float:
  return 0.0 if abs(v) < dz else v


def main() -> int:
  ap = argparse.ArgumentParser(description=__doc__)
  ap.add_argument("--device", default=None, help="Joystick device (default: first /dev/input/js*)")
  ap.add_argument("--lx", type=int, default=0, help="Left stick X axis index")
  ap.add_argument("--ly", type=int, default=1, help="Left stick Y axis index")
  ap.add_argument("--rx", type=int, default=3, help="Right stick X axis index")
  ap.add_argument("--ry", type=int, default=4, help="Right stick Y axis index")
  ap.add_argument("--deadzone", type=float, default=0.08, help="Stick deadzone")
  ap.add_argument("--invert-ly", action="store_true", default=True, help="Invert left Y (default on)")
  ap.add_argument("--no-invert-ly", action="store_false", dest="invert_ly")
  ap.add_argument("--invert-ry", action="store_true", default=True, help="Invert right Y (default on)")
  ap.add_argument("--no-invert-ry", action="store_false", dest="invert_ry")
  ap.add_argument("--hz", type=float, default=30.0, help="Display refresh rate")
  args = ap.parse_args()

  device = _find_device(args.device)
  fd = os.open(device, os.O_RDONLY | os.O_NONBLOCK)

  axes: dict[int, float] = {}
  buttons: dict[int, int] = {}
  period = 1.0 / max(args.hz, 1.0)
  last_draw = 0.0

  print(f"Device: {device}")
  print(
    f"Mapping: L=({args.lx},{args.ly})  R=({args.rx},{args.ry})  "
    f"deadzone={args.deadzone}"
  )
  print("Move the sticks. Ctrl+C to quit.\n")

  try:
    while True:
      # Drain events.
      while True:
        r, _, _ = select.select([fd], [], [], 0.0)
        if not r:
          break
        try:
          data = os.read(fd, _JS_EVENT.size)
        except BlockingIOError:
          break
        if len(data) < _JS_EVENT.size:
          break
        _t, value, typ, number = _JS_EVENT.unpack(data)
        etype = typ & ~_JS_EVENT_INIT
        if etype == _JS_EVENT_AXIS:
          axes[number] = value / 32767.0
        elif etype == _JS_EVENT_BUTTON:
          buttons[number] = int(value)

      now = time.time()
      if now - last_draw < period:
        time.sleep(0.001)
        continue
      last_draw = now

      lx = _deadzone(axes.get(args.lx, 0.0), args.deadzone)
      ly = _deadzone(axes.get(args.ly, 0.0), args.deadzone)
      rx = _deadzone(axes.get(args.rx, 0.0), args.deadzone)
      ry = _deadzone(axes.get(args.ry, 0.0), args.deadzone)
      if args.invert_ly:
        ly = -ly
      if args.invert_ry:
        ry = -ry

      # Clear screen + redraw (works in gnome-terminal / xterm).
      sys.stdout.write("\033[H\033[J")
      sys.stdout.write("=== Joystick stick monitor ===\n")
      sys.stdout.write(f"device: {device}\n\n")
      sys.stdout.write("LEFT STICK\n")
      sys.stdout.write(f"  X {lx:+.3f}  {_bar(lx)}\n")
      sys.stdout.write(f"  Y {ly:+.3f}  {_bar(ly)}   (up = + after invert)\n\n")
      sys.stdout.write("RIGHT STICK\n")
      sys.stdout.write(f"  X {rx:+.3f}  {_bar(rx)}\n")
      sys.stdout.write(f"  Y {ry:+.3f}  {_bar(ry)}   (up = + after invert)\n\n")
      sys.stdout.write(
        f"raw axes: "
        + " ".join(f"{i}:{axes.get(i, 0.0):+.2f}" for i in sorted(axes))
        + "\n"
      )
      pressed = [str(i) for i, v in sorted(buttons.items()) if v]
      sys.stdout.write(
        "buttons pressed: " + (", ".join(pressed) if pressed else "(none)") + "\n"
      )
      sys.stdout.write(
        "\nIf L/R sticks look wrong, remapping tip:\n"
        "  Xbox-style (default): --lx 0 --ly 1 --rx 3 --ry 4\n"
        "  Some pads:            --lx 0 --ly 1 --rx 2 --ry 3\n"
      )
      sys.stdout.flush()
  except KeyboardInterrupt:
    print("\nBye.")
  finally:
    os.close(fd)
  return 0


if __name__ == "__main__":
  raise SystemExit(main())
