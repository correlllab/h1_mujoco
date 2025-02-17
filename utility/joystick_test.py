import pygame
import struct
import time

pygame.init()
pygame.joystick.init()
joystick_count = pygame.joystick.get_count()
joystick = None
if joystick_count > 0:
    joystick = pygame.joystick.Joystick(0)
    joystick.init()
else:
    print("gg no joystick")

axis_id = {
    "LX": 0,  # Left stick axis x
    "LY": 1,  # Left stick axis y
    "RX": 3,  # Right stick axis x
    "RY": 4,  # Right stick axis y
    "LT": 2,  # Left trigger
    "RT": 5,  # Right trigger
    "DX": 6,  # Directional pad x
    "DY": 7,  # Directional pad y
}

button_id = {
    "X": 2,
    "Y": 3,
    "B": 1,
    "A": 0,
    "LB": 4,
    "RB": 5,
    "SELECT": 6,
    "START": 7,
}

key_map = {
    "R1": 0,
    "L1": 1,
    "start": 2,
    "select": 3,
    "R2": 4,
    "L2": 5,
    "F1": 6,
    "F2": 7,
    "A": 8,
    "B": 9,
    "X": 10,
    "Y": 11,
    "up": 12,
    "right": 13,
    "down": 14,
    "left": 15,
}

def print_lowstate():
    print(int(
        "".join(
            [
                f"{key}"
                for key in [
                    0,
                    0,
                    int(joystick.get_axis(axis_id["LT"]) > 0),
                    int(joystick.get_axis(axis_id["RT"]) > 0),
                    int(joystick.get_button(button_id["SELECT"])),
                    int(joystick.get_button(button_id["START"])),
                    int(joystick.get_button(button_id["LB"])),
                    int(joystick.get_button(button_id["RB"])),
                ]
            ]
        ),
        2,
    ))
    print(int(
        "".join(
            [
                f"{key}"
                for key in [
                    int(joystick.get_hat(0)[0] < 0),  # left
                    int(joystick.get_hat(0)[1] < 0),  # down
                    int(joystick.get_hat(0)[0] > 0), # right
                    int(joystick.get_hat(0)[1] > 0),    # up
                    int(joystick.get_button(button_id["Y"])),     # Y
                    int(joystick.get_button(button_id["X"])),     # X
                    int(joystick.get_button(button_id["B"])),     # B
                    int(joystick.get_button(button_id["A"])),     # A
                ]
            ]
        ),
        2,
    ))
    # Axes
    sticks = [
        joystick.get_axis(axis_id["LX"]),
        joystick.get_axis(axis_id["RX"]),
        -joystick.get_axis(axis_id["RY"]),
        -joystick.get_axis(axis_id["LY"]),
    ]
    packs = list(map(lambda x: struct.pack("f", x), sticks))

    print(packs[0])
    print(packs[1])
    print(packs[2])
    print(packs[3])

def print_highstate():
    pygame.event.get()
    key_state = [0] * 16
    key_state[key_map["R1"]] = joystick.get_button(
        button_id["RB"]
    )
    key_state[key_map["L1"]] = joystick.get_button(
        button_id["LB"]
    )
    key_state[key_map["start"]] = joystick.get_button(
        button_id["START"]
    )
    key_state[key_map["select"]] = joystick.get_button(
        button_id["SELECT"]
    )
    key_state[key_map["R2"]] = (
        joystick.get_axis(axis_id["RT"]) > 0
    )
    key_state[key_map["L2"]] = (
        joystick.get_axis(axis_id["LT"]) > 0
    )
    key_state[key_map["F1"]] = 0
    key_state[key_map["F2"]] = 0
    key_state[key_map["A"]] = joystick.get_button(button_id["A"])
    key_state[key_map["B"]] = joystick.get_button(button_id["B"])
    key_state[key_map["X"]] = joystick.get_button(button_id["X"])
    key_state[key_map["Y"]] = joystick.get_button(button_id["Y"])
    key_state[key_map["up"]] = joystick.get_hat(0)[1] > 0
    key_state[key_map["right"]] = joystick.get_hat(0)[0] > 0
    key_state[key_map["left"]] = joystick.get_hat(0)[0] < 0

    key_value = 0
    for i in range(16):
        key_value += key_state[i] << i

    print(key_value)

    print(joystick.get_axis(axis_id["LX"]))
    print(-joystick.get_axis(axis_id["LY"]))
    print(joystick.get_axis(axis_id["RX"]))
    print(-joystick.get_axis(axis_id["RY"]))

for _ in range(100):
    pygame.event.get()
    print('low state')
    print_lowstate()

    print('high state')
    print_highstate()

    time.sleep(0.1)

