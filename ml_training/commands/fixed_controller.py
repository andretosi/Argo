from dataclasses import dataclass

from commands.device import Device, DeviceConfig, Input


class FixedController(Device):
    """
    A constant command, used to:
    - train an agent to stand still (mode "still": all zeros)
    - train an agent to walk forward at a constant pace (mode "walk": only vx is non-zero)
    """

    name = "fixed"

    @dataclass
    class Params(DeviceConfig):
        mode: str = "still"  # "still": always zero; "walk": constant 0.2 forward

    def __init__(self, mode: "str | FixedController.Params" = "walk") -> None:
        if isinstance(mode, FixedController.Params):
            mode = mode.mode
        if mode in ["still", "walk"]:
            self.mode = mode
        else:
            raise ValueError(f"Unknown mode requested ({mode})")

        if self.mode == "still":
            self.i = Input(0, 0, 0)
        elif self.mode == "walk":
            self.i = Input(0.2, 0, 0)

    def update(self) -> None:
        return

    def read(self) -> Input:
        return self.i

    def reset(self) -> None:
        return
