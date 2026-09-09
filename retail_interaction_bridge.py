"""
Project-local bridge for the existing interaction_tracker.py.

This file intentionally does NOT depend on the external DeepCamera repository.
It adapts the user's existing interaction tracker to the retail sales pipeline.
"""

from __future__ import annotations

import importlib
from typing import Any, Dict


class InteractionTrackerBridge:
    def __init__(self, *args, **kwargs):
        self._tracker = self._build_tracker(*args, **kwargs)

    @staticmethod
    def _build_tracker(*args, **kwargs):
        try:
            module = importlib.import_module("interaction_tracker")
        except ImportError:
            # Keep the main application importable even if the optional
            # interaction tracker file has not been added yet.
            return None

        for class_name in (
            "InteractionTracker",
            "PersonObjectInteractionTracker",
            "ObjectInteractionTracker",
        ):
            cls = getattr(module, class_name, None)

            if cls is None:
                continue

            try:
                return cls(*args, **kwargs)

            except TypeError:
                try:
                    return cls()
                except Exception:
                    continue

            except Exception:
                continue

        return None

    @staticmethod
    def _call_update(tracker, tracks, objects, frame, now):
        if tracker is None:
            return {}

        candidates = (
            "update",
            "process",
            "associate",
            "track",
        )

        for name in candidates:
            fn = getattr(tracker, name, None)

            if not callable(fn):
                continue

            # First try the keyword API used by the retail pipeline.
            try:
                return fn(
                    tracks=tracks,
                    objects=objects,
                    frame=frame,
                    now=now,
                )

            except TypeError:
                pass

            # Try common positional forms used by interaction trackers.
            for args in (
                (tracks, objects, frame, now),
                (tracks, objects, frame),
                (tracks, objects),
                (tracks,),
            ):
                try:
                    return fn(*args)

                except TypeError:
                    continue

                except Exception:
                    return {}

        return {}

    @staticmethod
    def _person_id(item, fallback=None):
        if isinstance(item, dict):
            for key in (
                "person_id",
                "track_id",
                "id",
                "person",
            ):
                if key in item:
                    return item[key]

        for key in (
            "person_id",
            "track_id",
            "id",
        ):
            value = getattr(item, key, None)

            if value is not None:
                return value

        return fallback

    @staticmethod
    def _bool_value(item, *keys):
        if isinstance(item, dict):
            for key in keys:
                if key in item:
                    return bool(item[key])

        else:
            for key in keys:
                value = getattr(item, key, None)

                if value is not None:
                    return bool(value)

        return False

    def _normalize(self, result) -> Dict[Any, Dict[str, Any]]:
        if result is None:
            return {}

        if isinstance(result, dict):
            iterable = result.items()

        elif isinstance(result, (list, tuple)):
            iterable = enumerate(result)

        else:
            return {}

        normalized = {}

        for key, value in iterable:
            pid = self._person_id(
                value,
                fallback=key,
            )

            normalized[pid] = {
                "product_interaction": self._bool_value(
                    value,
                    "product_interaction",
                    "interacting_with_product",
                    "product_touch",
                    "product_interacted",
                    "interaction",
                ),

                "phone_comparison": self._bool_value(
                    value,
                    "phone_comparison",
                    "comparing_phone",
                    "phone_interaction",
                    "phone_compare",
                ),

                "details": value,
            }

        return normalized

    def update(
        self,
        tracks,
        objects,
        frame=None,
        now=None,
    ):
        result = self._call_update(
            self._tracker,
            tracks,
            objects,
            frame,
            now,
        )

        return self._normalize(result)

    def process(
        self,
        tracks,
        objects,
        frame=None,
        now=None,
    ):
        return self.update(
            tracks,
            objects,
            frame,
            now,
        )

    def status(self):
        """
        Return a human-readable status for main.py.

        If the underlying interaction_tracker.py provides its own
        status() method, use that. Otherwise report the bridge state.
        """
        if self._tracker is None:
            return "Interaction tracker: unavailable"

        status_fn = getattr(
            self._tracker,
            "status",
            None,
        )

        if callable(status_fn):
            try:
                return status_fn()
            except Exception:
                pass

        return "Interaction tracker: ready"

    def close(self):
        if self._tracker is None:
            return

        for name in (
            "close",
            "stop",
            "shutdown",
        ):
            fn = getattr(
                self._tracker,
                name,
                None,
            )

            if callable(fn):
                try:
                    fn()
                except Exception:
                    pass

                break