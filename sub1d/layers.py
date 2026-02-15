"""Stratigraphic layer domain model for the SUB1D subsidence model.

This module provides the :class:`Stratigraphy` class, which encapsulates the
ordered sequence of aquifer and aquitard layers, their compaction behaviour,
interbed configuration, and the spatial relationships between adjacent layers.
It serves as the single source of truth for layer-related queries used
throughout the simulation pipeline.

Typical usage
-------------
Construct directly::

    strat = Stratigraphy(
        layer_names=["Upper Aquifer", "Corcoran Clay", "Lower Aquifer"],
        layer_types={"Upper Aquifer": "Aquifer",
                     "Corcoran Clay": "Aquitard",
                     "Lower Aquifer": "Aquifer"},
        layer_compaction_switch={"Upper Aquifer": True,
                                 "Corcoran Clay": True,
                                 "Lower Aquifer": False},
        interbeds_switch={"Upper Aquifer": True,
                          "Corcoran Clay": False,
                          "Lower Aquifer": False},
    )

Or from a ``ModelConfig`` object::

    strat = Stratigraphy.from_config(config)
"""

from __future__ import annotations

import logging
from typing import Optional

logger = logging.getLogger(__name__)


class Stratigraphy:
    """Encapsulates the stratigraphic layer model and relationships.

    The class stores an ordered list of layer names together with metadata
    that describes each layer's type (``"Aquifer"`` or ``"Aquitard"``),
    whether it participates in compaction, whether it contains interbeds,
    and -- when interbeds are present -- their thickness distribution
    parameters.

    Parameters
    ----------
    layer_names : list[str]
        Ordered sequence of layer names from top (shallowest) to bottom
        (deepest).
    layer_types : dict[str, str]
        Mapping of each layer name to its type.  Accepted values are
        ``"Aquifer"`` and ``"Aquitard"``.
    layer_compaction_switch : dict[str, bool]
        Mapping of each layer name to a boolean indicating whether the
        layer participates in compaction calculations.
    interbeds_switch : dict[str, bool]
        Mapping of each layer name to a boolean indicating whether the
        layer contains interbedded clay lenses.
    interbeds_distributions : dict[str, dict] or None, optional
        Mapping of layer names to dictionaries describing the clay-bed
        thickness distribution parameters (e.g. ``{"distribution": "uniform",
        "thickness": 1.5}``).  Only required for layers where interbeds are
        enabled.  Defaults to ``None`` (empty).

    Raises
    ------
    ValueError
        If any layer in *layer_names* is missing from *layer_types*, or if
        a layer type is not one of the two accepted values.

    Examples
    --------
    >>> strat = Stratigraphy(
    ...     layer_names=["Aquifer_1", "Aquitard_1", "Aquifer_2"],
    ...     layer_types={"Aquifer_1": "Aquifer",
    ...                  "Aquitard_1": "Aquitard",
    ...                  "Aquifer_2": "Aquifer"},
    ...     layer_compaction_switch={"Aquifer_1": True,
    ...                              "Aquitard_1": True,
    ...                              "Aquifer_2": False},
    ...     interbeds_switch={"Aquifer_1": False,
    ...                       "Aquitard_1": False,
    ...                       "Aquifer_2": False},
    ... )
    >>> strat.aquifer_names
    ['Aquifer_1', 'Aquifer_2']
    >>> strat.aquitard_names
    ['Aquitard_1']
    """

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    def __init__(
        self,
        layer_names: list[str],
        layer_types: dict[str, str],
        layer_compaction_switch: dict[str, bool],
        interbeds_switch: dict[str, bool],
        interbeds_distributions: dict[str, dict] | None = None,
    ) -> None:
        self._layer_names: list[str] = list(layer_names)
        self._layer_types: dict[str, str] = dict(layer_types)
        self._layer_compaction_switch: dict[str, bool] = dict(layer_compaction_switch)
        self._interbeds_switch: dict[str, bool] = dict(interbeds_switch)
        self._interbeds_distributions: dict[str, dict] = dict(
            interbeds_distributions or {}
        )
        self._validate()
        logger.info(
            "Stratigraphy initialised with %d layers (%d aquifers, %d aquitards).",
            len(self._layer_names),
            len(self.aquifer_names),
            len(self.aquitard_names),
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _validate(self) -> None:
        """Check that layer types are valid and every name has metadata.

        Raises
        ------
        ValueError
            If a layer listed in *layer_names* is absent from
            *layer_types*, or if the type string is not ``"Aquifer"`` or
            ``"Aquitard"``.
        """
        valid_types = {"Aquifer", "Aquitard"}
        for name in self._layer_names:
            if name not in self._layer_types:
                raise ValueError(
                    f"Layer '{name}' is listed in layer_names but missing "
                    f"from layer_types."
                )
            ltype = self._layer_types[name]
            if ltype not in valid_types:
                raise ValueError(
                    f"Layer '{name}' has invalid type '{ltype}'. "
                    f"Accepted values are {valid_types}."
                )
        logger.debug("Stratigraphy validation passed for all %d layers.", len(self._layer_names))

    # ------------------------------------------------------------------
    # Layer name queries
    # ------------------------------------------------------------------

    @property
    def layer_names(self) -> list[str]:
        """Return an ordered list of all layer names (top to bottom).

        Returns
        -------
        list[str]
            A *copy* of the internal layer-name list so that callers
            cannot mutate the stratigraphy.
        """
        return list(self._layer_names)

    @property
    def aquifer_names(self) -> list[str]:
        """Return the names of all aquifer layers, preserving order.

        Returns
        -------
        list[str]
            Layer names whose type is ``"Aquifer"``, ordered from top to
            bottom.
        """
        return [n for n in self._layer_names if self._layer_types[n] == "Aquifer"]

    @property
    def aquitard_names(self) -> list[str]:
        """Return the names of all aquitard layers, preserving order.

        Returns
        -------
        list[str]
            Layer names whose type is ``"Aquitard"``, ordered from top to
            bottom.
        """
        return [n for n in self._layer_names if self._layer_types[n] == "Aquitard"]

    @property
    def compactable_aquifer_names(self) -> list[str]:
        """Return aquifer layers that have compaction enabled.

        An aquifer compacts when it contains compressible sand or fine-grained
        material and its ``layer_compaction_switch`` is ``True``.

        Returns
        -------
        list[str]
            Subset of :attr:`aquifer_names` with compaction enabled.
        """
        return [
            n for n in self.aquifer_names
            if self._layer_compaction_switch.get(n, False)
        ]

    @property
    def interbedded_layer_names(self) -> list[str]:
        """Return layers that contain clay interbeds.

        Returns
        -------
        list[str]
            Layer names (aquifers or aquitards) whose ``interbeds_switch``
            is ``True``.
        """
        return [n for n, v in self._interbeds_switch.items() if v]

    @property
    def layers_requiring_solving(self) -> list[str]:
        """Return layers that require the compaction solver.

        These are all interbedded layers (which need diffusion-based head
        propagation through their clay lenses) plus all aquitard layers
        (which are always solved for vertical head diffusion).

        Returns
        -------
        list[str]
            Concatenation of :attr:`interbedded_layer_names` and
            :attr:`aquitard_names`.

        Notes
        -----
        A layer may appear in both lists (e.g. an interbedded aquitard).
        Downstream consumers should handle potential duplicates if
        necessary.
        """
        return self.interbedded_layer_names + self.aquitard_names

    # ------------------------------------------------------------------
    # Adjacency queries
    # ------------------------------------------------------------------

    def aquifer_above(self, aquitard: str) -> Optional[str]:
        """Return the aquifer immediately above an aquitard, if any.

        Parameters
        ----------
        aquitard : str
            Name of the aquitard layer to query.

        Returns
        -------
        str or None
            The name of the aquifer directly above *aquitard* in the
            layer stack, or ``None`` if *aquitard* is the topmost layer
            or the layer above it is not an aquifer.

        Raises
        ------
        ValueError
            If *aquitard* is not found in the layer stack (raised by
            ``list.index``).
        """
        idx: int = self._layer_names.index(aquitard)
        if idx > 0 and self._layer_types[self._layer_names[idx - 1]] == "Aquifer":
            return self._layer_names[idx - 1]
        return None

    def aquifer_below(self, aquitard: str) -> Optional[str]:
        """Return the aquifer immediately below an aquitard, if any.

        Parameters
        ----------
        aquitard : str
            Name of the aquitard layer to query.

        Returns
        -------
        str or None
            The name of the aquifer directly below *aquitard* in the
            layer stack, or ``None`` if *aquitard* is the bottommost
            layer or the layer below it is not an aquifer.

        Raises
        ------
        ValueError
            If *aquitard* is not found in the layer stack (raised by
            ``list.index``).
        """
        idx: int = self._layer_names.index(aquitard)
        if (
            idx < len(self._layer_names) - 1
            and self._layer_types[self._layer_names[idx + 1]] == "Aquifer"
        ):
            return self._layer_names[idx + 1]
        return None

    @property
    def aquitard_positions(self) -> dict[str, int]:
        """Map each aquitard name to its zero-based position in the stack.

        Returns
        -------
        dict[str, int]
            Keys are aquitard names; values are integer indices into
            :attr:`layer_names`.
        """
        return {
            n: i
            for i, n in enumerate(self._layer_names)
            if self._layer_types[n] == "Aquitard"
        }

    @property
    def aquifers_above_aquitards(self) -> list[str]:
        """Return aquifers that sit directly above any aquitard.

        These aquifers supply the upper boundary condition for the
        vertical head-diffusion equation within their underlying
        aquitard.

        Returns
        -------
        list[str]
            Unique aquifer names (order follows aquitard iteration).
        """
        result: list[str] = []
        for aqt in self.aquitard_names:
            above: Optional[str] = self.aquifer_above(aqt)
            if above is not None:
                result.append(above)
        return result

    @property
    def aquifers_below_aquitards(self) -> list[str]:
        """Return aquifers that sit directly below any aquitard.

        These aquifers supply the lower boundary condition for the
        vertical head-diffusion equation within their overlying
        aquitard.

        Returns
        -------
        list[str]
            Unique aquifer names (order follows aquitard iteration).
        """
        result: list[str] = []
        for aqt in self.aquitard_names:
            below: Optional[str] = self.aquifer_below(aqt)
            if below is not None:
                result.append(below)
        return result

    @property
    def all_aquifers_needing_head_data(self) -> list[str]:
        """Return every aquifer that requires externally supplied head data.

        An aquifer needs head input if it:

        * Borders an aquitard from above (upper boundary condition),
        * Borders an aquitard from below (lower boundary condition), or
        * Has its own compaction enabled (sand-skeleton compaction).

        Returns
        -------
        list[str]
            De-duplicated list of aquifer names (order is *not*
            guaranteed because a ``set`` is used internally for
            de-duplication).
        """
        return list(
            set(
                self.aquifers_above_aquitards
                + self.aquifers_below_aquitards
                + self.compactable_aquifer_names
            )
        )

    # ------------------------------------------------------------------
    # Single-layer queries
    # ------------------------------------------------------------------

    def layer_type(self, name: str) -> str:
        """Return the type string for a given layer.

        Parameters
        ----------
        name : str
            Layer name.

        Returns
        -------
        str
            Either ``"Aquifer"`` or ``"Aquitard"``.

        Raises
        ------
        KeyError
            If *name* is not a recognised layer.
        """
        return self._layer_types[name]

    def is_aquifer(self, name: str) -> bool:
        """Check whether a layer is an aquifer.

        Parameters
        ----------
        name : str
            Layer name.

        Returns
        -------
        bool
            ``True`` if the layer's type is ``"Aquifer"``.

        Raises
        ------
        KeyError
            If *name* is not a recognised layer.
        """
        return self._layer_types[name] == "Aquifer"

    def is_aquitard(self, name: str) -> bool:
        """Check whether a layer is an aquitard.

        Parameters
        ----------
        name : str
            Layer name.

        Returns
        -------
        bool
            ``True`` if the layer's type is ``"Aquitard"``.

        Raises
        ------
        KeyError
            If *name* is not a recognised layer.
        """
        return self._layer_types[name] == "Aquitard"

    def get_interbeds_distribution(self, layer: str) -> dict:
        """Return the interbed thickness distribution for a layer.

        Parameters
        ----------
        layer : str
            Layer name.

        Returns
        -------
        dict
            Distribution parameters (e.g. ``{"distribution": "uniform",
            "thickness": 1.5}``).  Returns an empty ``dict`` when no
            distribution has been specified for the layer.
        """
        return self._interbeds_distributions.get(layer, {})

    def has_interbeds(self, layer: str) -> bool:
        """Check whether a layer contains clay interbeds.

        Parameters
        ----------
        layer : str
            Layer name.

        Returns
        -------
        bool
            ``True`` if the layer's ``interbeds_switch`` is ``True``.
        """
        return self._interbeds_switch.get(layer, False)

    # ------------------------------------------------------------------
    # Alternate constructors
    # ------------------------------------------------------------------

    @classmethod
    def from_config(cls, config) -> Stratigraphy:
        """Construct a :class:`Stratigraphy` from a ``ModelConfig`` object.

        This factory method extracts the relevant layer metadata from a
        ``ModelConfig`` instance and delegates to the primary constructor.

        Parameters
        ----------
        config : ModelConfig
            A configuration object that exposes at minimum the following
            attributes:

            * ``layer_names`` -- ``list[str]``
            * ``layer_types`` -- ``dict[str, str]``
            * ``layers`` -- iterable of layer-config objects, each having
              ``name`` (``str``), ``compaction_switch`` (``bool``),
              ``interbeds_switch`` (``bool``), and optionally
              ``interbeds_distributions`` (``dict`` or ``None``).

        Returns
        -------
        Stratigraphy
            A fully initialised stratigraphy instance.

        Examples
        --------
        >>> strat = Stratigraphy.from_config(my_model_config)
        >>> strat.layer_names
        ['Upper Aquifer', 'Corcoran Clay', 'Lower Aquifer']
        """
        logger.info("Building Stratigraphy from ModelConfig.")
        return cls(
            layer_names=config.layer_names,
            layer_types=config.layer_types,
            layer_compaction_switch={
                lc.name: lc.compaction_switch for lc in config.layers
            },
            interbeds_switch={
                lc.name: lc.interbeds_switch for lc in config.layers
            },
            interbeds_distributions={
                lc.name: lc.interbeds_distributions
                for lc in config.layers
                if lc.interbeds_distributions
            },
        )

    # ------------------------------------------------------------------
    # Dunder methods
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        """Return an unambiguous string representation.

        Returns
        -------
        str
            A string of the form
            ``Stratigraphy(n_layers=5, aquifers=3, aquitards=2)``.
        """
        return (
            f"Stratigraphy(n_layers={len(self._layer_names)}, "
            f"aquifers={len(self.aquifer_names)}, "
            f"aquitards={len(self.aquitard_names)})"
        )

    def __len__(self) -> int:
        """Return the total number of layers.

        Returns
        -------
        int
            Number of layers in the stratigraphy.
        """
        return len(self._layer_names)
