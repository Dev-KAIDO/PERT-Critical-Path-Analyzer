"""
Navigation constants for the application pages.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum


class NavDestination(IntEnum):
    """Destinations in the navigation bar."""

    ANALYZE = 0
    UNDERSTANDING = 1
    REVIEW = 2
    VALIDATE = 3
    RESULTS = 4
    NETWORK_BUILDER = 5


@dataclass(frozen=True)
class NavItem:
    """Descriptor for a navigation destination."""

    destination: NavDestination
    label: str
    icon: str = ""
    key: str = ""

    def __post_init__(self) -> None:
        if not self.key:
            object.__setattr__(self, "key", self.destination.name.lower())


NAV_ITEMS: list[NavItem] = [
    NavItem(NavDestination.ANALYZE, "Analyze", icon="\u2315"),
    NavItem(NavDestination.UNDERSTANDING, "Understand", icon="\u25C8"),
    NavItem(NavDestination.REVIEW, "Review", icon="\u270E"),
    NavItem(NavDestination.RESULTS, "Results", icon="\u25A3"),
    NavItem(NavDestination.NETWORK_BUILDER, "Build Network", icon="\u271A"),
]
