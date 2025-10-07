from __future__ import annotations
from typing import Any, Dict, Type

from .base import BaseLoss
from .vanilla import VanillaLoss
from .c51 import C51Loss
from .qr import QRLoss
from .munchausen import MunchausenLoss, MunchausenC51Loss, MunchausenQRLoss


class LossFactory:
    """
    LossFactory — a factory class for creating loss function instances
    based on a string identifier.

    Supported `loss_name` values:
        - `"vanilla"`          → :class:`VanillaLoss`
        - `"c51"`              → :class:`C51Loss`
        - `"qr"`               → :class:`QRLoss`
        - `"munchausen"`       → :class:`MunchausenLoss`
        - `"munchausen_c51"`   → :class:`MunchausenC51Loss`
        - `"munchausen_qr"`    → :class:`MunchausenQRLoss`

    Notes
    -----
    - All loss classes must inherit from :class:`BaseLoss`.
    - Parameters are forwarded to the target class constructor via ``**params``.
    """

    _registry: Dict[str, Type[BaseLoss]] = {
        "vanilla": VanillaLoss,
        "c51": C51Loss,
        "qr": QRLoss,
        "munchausen": MunchausenLoss,
        "munchausen_c51": MunchausenC51Loss,
        "munchausen_qr": MunchausenQRLoss,
    }

    @staticmethod
    def create(loss_name: str, params: Dict[str, Any]) -> BaseLoss:
        """
        Create a loss instance by name.

        Parameters
        ----------
        loss_name : str
            The name of the loss function (case-insensitive).
        params : dict
            Parameters to be passed to the loss class constructor.

        Returns
        -------
        BaseLoss
            An initialized loss class instance.

        Raises
        ------
        ValueError
            If the specified loss name is not recognized.
        """
        name = loss_name.lower()
        if name not in LossFactory._registry:
            raise ValueError(
                f"Unknown loss name: {loss_name!r}. "
                f"Available: {list(LossFactory._registry.keys())}"
            )
        return LossFactory._registry[name](**params)
