"""A 3DConnexion SpaceMouse as a lerobot teleoperator, emitting per-frame increments.

The cap's six axes are routed to the five the arm has: three translate the tool, one
tilts it, one turns the jaw. The two cap buttons work the gripper.

`pyspacemouse.read()` drains a few HID packets per call, so a consumer polling at the
control rate falls behind the device's kilohertz packet rate while the cap is moving and
reads stale state. A reader thread drains it continuously and `get_action` takes the
latest.
"""

from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass, field
from types import SimpleNamespace
from typing import Any, ClassVar

from lerobot.teleoperators.config import TeleoperatorConfig
from lerobot.teleoperators.teleoperator import Teleoperator
from lerobot.types import RobotAction
from lerobot.utils.decorators import check_if_already_connected, check_if_not_connected
from lerobot.utils.import_utils import require_package

logger = logging.getLogger(__name__)

_CENTRED = SimpleNamespace(x=0.0, y=0.0, z=0.0, roll=0.0, pitch=0.0, yaw=0.0, buttons=[0, 0])


@TeleoperatorConfig.register_subclass("vla_test_spacemouse")
@dataclass
class SpaceMouseConfig(TeleoperatorConfig):
    """Rig-specific routing and gains.

    The cap's axes do not have to line up with the arm's, so remap and flip until pushing
    forward, lifting and twisting each move the tool the way they look like they should.
    Sensitivities are per unit of full cap deflection per frame, at `hw.robot.FPS`.
    """

    # Which cap axis drives each of the five things the arm can do.
    routing: dict[str, str] = field(default_factory=lambda: {
        "dx": "y", "dy": "x", "dz": "z", "d_tilt": "pitch", "d_yaw": "yaw",
    })
    inverted: tuple[str, ...] = ("dy",)
    deadband: float = 0.05

    metres_per_frame: float = 0.0045
    radians_per_frame: float = 0.05

    open_button: int = 0
    close_button: int = 1

    device_path: str | None = None


class SpaceMouse(Teleoperator):
    """Per-frame increments: `dx`/`dy`/`dz` in metres, `d_tilt`/`d_yaw` in radians, and
    `gripper` in -1, 0, +1 for close, hold, open."""

    config_class: ClassVar[type] = SpaceMouseConfig
    name: ClassVar[str] = "vla_test_spacemouse"

    def __init__(self, config: SpaceMouseConfig):
        require_package("pyspacemouse", extra="spacemouse")
        super().__init__(config)
        self.config = config
        self._connected = False
        self._state = _CENTRED
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._reader: threading.Thread | None = None

    @property
    def action_features(self) -> dict[str, type]:
        return dict.fromkeys(["dx", "dy", "dz", "d_tilt", "d_yaw", "gripper"], float)

    @property
    def feedback_features(self) -> dict:
        return {}

    @property
    def is_connected(self) -> bool:
        return self._connected

    @property
    def is_calibrated(self) -> bool:
        return True

    def calibrate(self) -> None:
        """The cap centres itself in firmware."""

    def configure(self) -> None:
        pass

    def send_feedback(self, feedback: dict[str, Any]) -> None:
        pass

    @check_if_already_connected
    def connect(self, calibrate: bool = True) -> None:
        import pyspacemouse

        opened = pyspacemouse.open(**({"path": self.config.device_path}
                                      if self.config.device_path else {}))
        if not opened:
            raise ConnectionError(
                "pyspacemouse.open() failed. On Linux the SpaceMouse HID device needs a "
                "udev rule granting non-root access; see the pyspacemouse README."
            )
        self._connected = True
        self._state = _CENTRED
        self._stop.clear()
        self._reader = threading.Thread(target=self._drain, name="spacemouse", daemon=True)
        self._reader.start()

    @check_if_not_connected
    def disconnect(self) -> None:
        import pyspacemouse

        self._stop.set()
        if self._reader is not None:
            self._reader.join(timeout=1.0)
            self._reader = None
        try:
            pyspacemouse.close()
        finally:
            self._connected = False

    def _drain(self) -> None:
        import pyspacemouse

        while not self._stop.is_set():
            try:
                state = pyspacemouse.read()
            except Exception:
                logger.exception("pyspacemouse.read() failed, stopping the reader")
                break
            with self._lock:
                self._state = state
            time.sleep(0.001)

    def _axis(self, state, name):
        value = float(getattr(state, self.config.routing[name]))
        if abs(value) < self.config.deadband:
            return 0.0
        return -value if name in self.config.inverted else value

    @check_if_not_connected
    def get_action(self) -> RobotAction:
        with self._lock:
            state = self._state

        buttons = list(state.buttons or [])
        def pressed(index):
            return 0 <= index < len(buttons) and bool(buttons[index])

        return {
            **{name: self._axis(state, name) * self.config.metres_per_frame
               for name in ("dx", "dy", "dz")},
            **{name: self._axis(state, name) * self.config.radians_per_frame
               for name in ("d_tilt", "d_yaw")},
            "gripper": float(pressed(self.config.open_button)
                             - pressed(self.config.close_button)),
        }
