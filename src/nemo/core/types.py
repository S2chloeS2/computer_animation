from __future__ import annotations

from enum import IntEnum
from typing import Any, Literal

import numpy as np

try:
    from typing import override
except ImportError:
    try:
        from typing import override
    except ImportError:
        # Fallback no-op decorator if override is not available
        def override(func):
            return func


Vec2 = list[float] | tuple[float, float] | np.ndarray[Any, np.dtype[np.float64]]
"""A 2D vector represented as a list or tuple of 2 floats."""
Vec3 = list[float] | tuple[float, float, float] | np.ndarray[Any, np.dtype[np.float64]]
"""A 3D vector represented as a list or tuple of 3 floats."""

# type alias for numpy arrays
nparray = np.ndarray[Any, np.dtype[Any]]


class Axis(IntEnum):
    """Enumeration of axes in 3D space."""

    X = 0
    """X-axis."""
    Y = 1
    """Y-axis."""
    Z = 2
    """Z-axis."""

    @classmethod
    def from_string(cls, axis_str: str) -> Axis:
        """
        Convert a string representation of an axis ("x", "y", or "z") to the corresponding Axis enum member.

        Args:
            axis_str (str): The axis as a string. Should be "x", "y", or "z" (case-insensitive).

        Returns:
            Axis: The corresponding Axis enum member.

        Raises:
            ValueError: If the input string does not correspond to a valid axis.
        """
        axis_str = axis_str.lower()
        if axis_str == "x":
            return cls.X
        elif axis_str == "y":
            return cls.Y
        elif axis_str == "z":
            return cls.Z
        raise ValueError(f"Invalid axis string: {axis_str}")

    @classmethod
    def from_any(cls, value: AxisType) -> Axis:
        """
        Convert a value of various types to an Axis enum member.

        Args:
            value (AxisType): The value to convert. Can be an Axis, str, or int-like.

        Returns:
            Axis: The corresponding Axis enum member.

        Raises:
            TypeError: If the value cannot be converted to an Axis.
            ValueError: If the string or integer does not correspond to a valid Axis.
        """
        if isinstance(value, cls):
            return value
        if isinstance(value, str):
            return cls.from_string(value)
        if isinstance(value, int):
            return cls(value)
        raise TypeError(f"Cannot convert {type(value)} to Axis")

    @override
    def __str__(self):
        return self.name.capitalize()

    @override
    def __repr__(self):
        return f"Axis.{self.name.capitalize()}"

    @override
    def __eq__(self, other):
        if isinstance(other, str):
            return self.name.lower() == other.lower()
        if isinstance(other, int):
            return self.value == int(other)
        return NotImplemented

    @override
    def __hash__(self):
        return hash(self.name)

    def to_vector(self) -> tuple[float, float, float]:
        """
        Return the axis as a 3D unit vector.

        Returns:
            tuple[float, float, float]: The unit vector corresponding to the axis.
        """
        if self == Axis.X:
            return (1.0, 0.0, 0.0)
        elif self == Axis.Y:
            return (0.0, 1.0, 0.0)
        else:
            return (0.0, 0.0, 1.0)


AxisType = Axis | Literal["X", "Y", "Z"] | Literal[0, 1, 2] | int | str
"""Type that can be used to represent an axis, including the enum, string, and integer representations."""
