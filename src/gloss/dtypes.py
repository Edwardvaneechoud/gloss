"""Precise column types.

    from gloss import Column, Int, Decimal, String, DateTime

    Column("return_id", Int(32))
    Column("refund", Decimal(18, 2))
    Column("reason", String(255))
    Column("returned_at", DateTime(tz="UTC"))

Ossie's datatypes are logical (``Integer``, ``Decimal``, ``String`` …) and carry no
width or precision. A Gloss type maps to the Ossie datatype *and* keeps the precise
physical type as its own fact (``physical_type``, e.g. ``"Int32"``, ``"Decimal(18,2)"``),
which exports under ``nonportable_extensions`` and survives a round trip.

Bare classes work too: ``Column("id", Int)`` means "an integer, width unspecified".
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import ClassVar

from .types import to_ossie_datatype


@dataclass(frozen=True)
class DType:
    """Base of the precise column types: ``ossie`` is the Ossie datatype, ``physical`` the precise one."""

    ossie_name: ClassVar[str] = "Opaque"

    @property
    def ossie(self) -> str:
        """The Ossie logical datatype."""
        return self.ossie_name

    @property
    def physical(self) -> str | None:
        """Precise type string, or None when nothing beyond the Ossie type is known."""
        return None

    def __str__(self) -> str:
        return self.physical or self.ossie


@dataclass(frozen=True)
class Int(DType):
    """An integer, signed unless ``unsigned``; ``bits`` is 8, 16, 32, 64 or 128, or None for any width."""

    bits: int | None = None
    unsigned: bool = False
    ossie_name: ClassVar[str] = "Integer"

    def __post_init__(self) -> None:
        if self.bits is not None and self.bits not in (8, 16, 32, 64, 128):
            raise ValueError(f"Int bits must be 8, 16, 32, 64 or 128, not {self.bits}")

    def __repr__(self) -> str:
        name = "UInt" if self.unsigned else "Int"
        return f"{name}({self.bits})" if self.bits else f"{name}()"

    @property
    def physical(self) -> str | None:
        if self.bits is None:
            return "UInt" if self.unsigned else None
        return f"{'UInt' if self.unsigned else 'Int'}{self.bits}"


@dataclass(frozen=True, repr=False)
class UInt(Int):
    """An unsigned integer; ``UInt()`` without a width still records that it is unsigned."""

    unsigned: bool = True


@dataclass(frozen=True)
class Float(DType):
    """A floating-point number of 16, 32 or 64 ``bits``, or None for any width."""

    bits: int | None = None
    ossie_name: ClassVar[str] = "Float"

    def __post_init__(self) -> None:
        if self.bits is not None and self.bits not in (16, 32, 64):
            raise ValueError(f"Float bits must be 16, 32 or 64, not {self.bits}")

    def __repr__(self) -> str:
        return f"Float({self.bits})" if self.bits else "Float()"

    @property
    def physical(self) -> str | None:
        return f"Float{self.bits}" if self.bits else None


@dataclass(frozen=True)
class Decimal(DType):
    """A fixed-point number: ``precision`` of 1 to 76 digits, and a ``scale`` that needs a precision."""

    precision: int | None = None
    scale: int | None = None
    ossie_name: ClassVar[str] = "Decimal"

    def __post_init__(self) -> None:
        if self.scale is not None and self.precision is None:
            raise ValueError("Decimal scale needs a precision: Decimal(18, 2)")
        if self.precision is not None and not 1 <= self.precision <= 76:
            raise ValueError("Decimal precision must be between 1 and 76")
        if self.scale is not None and not 0 <= self.scale <= self.precision:  # type: ignore[operator]
            raise ValueError("Decimal scale must be between 0 and the precision")

    def __repr__(self) -> str:
        args = [str(a) for a in (self.precision, self.scale) if a is not None]
        return f"Decimal({', '.join(args)})"

    @property
    def physical(self) -> str | None:
        if self.precision is None:
            return None
        return f"Decimal({self.precision},{self.scale})" if self.scale is not None else f"Decimal({self.precision})"


@dataclass(frozen=True)
class String(DType):
    """Text, with an optional maximum ``length``."""

    length: int | None = None
    ossie_name: ClassVar[str] = "String"

    def __post_init__(self) -> None:
        if self.length is not None and self.length < 1:
            raise ValueError("String length must be positive")

    def __repr__(self) -> str:
        return f"String({self.length})" if self.length else "String()"

    @property
    def physical(self) -> str | None:
        return f"String({self.length})" if self.length else None


@dataclass(frozen=True)
class Boolean(DType):
    """A true/false value."""

    ossie_name: ClassVar[str] = "Boolean"


@dataclass(frozen=True)
class Date(DType):
    """A calendar date without a time."""

    ossie_name: ClassVar[str] = "Date"


@dataclass(frozen=True)
class Time(DType):
    """A time of day without a date."""

    ossie_name: ClassVar[str] = "Time"


_UNITS = ("s", "ms", "us", "ns")


@dataclass(frozen=True)
class DateTime(DType):
    """``DateTime()`` has no time zone; ``DateTime(tz="UTC")`` maps to Ossie ``DateTimeTz``."""

    tz: str | bool | None = None  # a zone name, or True for "has a zone, unspecified"
    unit: str | None = None

    def __post_init__(self) -> None:
        if self.unit is not None and self.unit not in _UNITS:
            raise ValueError(f"DateTime unit must be one of {_UNITS}")

    def __repr__(self) -> str:
        args = [f"tz={self.tz!r}" if self.tz is not None else "", f"unit={self.unit!r}" if self.unit else ""]
        return f"DateTime({', '.join(a for a in args if a)})"

    @property
    def ossie(self) -> str:
        return "DateTimeTz" if self.tz else "DateTime"

    @property
    def physical(self) -> str | None:
        args = [a for a in (self.unit, f"tz={self.tz}" if isinstance(self.tz, str) else None) if a]
        return f"DateTime({', '.join(args)})" if args else None


@dataclass(frozen=True)
class Opaque(DType):
    """A type outside Ossie's vocabulary, kept by name: ``Opaque("geometry")``."""

    name: str = "Opaque"

    @property
    def physical(self) -> str | None:
        return None if self.name == "Opaque" else self.name


ALL: tuple[type[DType], ...] = (Int, UInt, Float, Decimal, String, Boolean, Date, Time, DateTime, Opaque)

_PARSE = [
    (re.compile(r"^(U?)Int(\d+)$"), lambda m: (UInt if m[1] else Int)(int(m[2]))),
    (re.compile(r"^UInt$"), lambda m: UInt()),
    (re.compile(r"^Float(\d+)$"), lambda m: Float(int(m[1]))),
    (re.compile(r"^Decimal\((\d+)(?:,\s*(\d+))?\)$"), lambda m: Decimal(int(m[1]), int(m[2]) if m[2] else None)),
    (re.compile(r"^String\((\d+)\)$"), lambda m: String(int(m[1]))),
    (
        re.compile(r"^DateTime\((?:(s|ms|us|ns))?(?:,\s*)?(?:tz=([^)]+))?\)$"),
        lambda m: DateTime(tz=m[2], unit=m[1]),
    ),
]

_LOGICAL = {"Integer": Int, "Float": Float, "Decimal": Decimal, "String": String, "Boolean": Boolean,
            "Date": Date, "Time": Time, "DateTime": DateTime, "Opaque": Opaque}


def parse(physical: str | None, logical: str | None = None) -> DType | None:
    """Rebuild a DType from a stored physical type (Gloss format) and/or Ossie datatype."""
    if physical:
        for rx, make in _PARSE:
            m = rx.match(physical)
            if m:
                return make(m)
        mapped = to_ossie_datatype(physical)
        if mapped and mapped != "Opaque" and (logical is None or mapped == logical):
            return _from_logical(mapped)
        return Opaque(physical)
    return _from_logical(logical) if logical else None


def _from_logical(logical: str) -> DType:
    if logical == "DateTimeTz":
        return DateTime(tz=True)
    return _LOGICAL[logical]()
