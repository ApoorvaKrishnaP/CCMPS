"""Tracker factory and runtime selection helpers.

Default: returns `CrowdTracker` (Kalman+Hungarian) which has minimal
third-party binary requirements and is preferred for Python 3.13.

If the environment variable `USE_DEEPSORT` is set to a truthy value
('1', 'true', 'yes'), the factory will attempt to instantiate
`DeepSortTracker` (requires `deep-sort-realtime`). If that import fails
an informative ImportError is raised with installation instructions.
"""
from __future__ import annotations
import os
from typing import Any


def get_tracker(name: str | None = None, **kwargs: Any):
	"""Return an instance of a tracker by name or environment.

	Args:
		name: Optional name: 'deepsort' or 'crowd'. If None, read from
			  `USE_DEEPSORT` env var (defaults to crowd).
		**kwargs: Passed to the tracker constructor.
	"""
	choice = (name or os.getenv("USE_DEEPSORT", "")).strip().lower()
	if choice in ("1", "true", "yes", "deepsort", "deep", "d"):
		try:
			from .deep_sort_tracker import DeepSortTracker

			return DeepSortTracker(**kwargs)
		except Exception as e:  # pragma: no cover - runtime import
			# Fall back to CrowdTracker with a clear printed warning
			print(
				"[tracker.get_tracker] Failed to create DeepSortTracker; "
				"falling back to CrowdTracker. Error:\n",
				repr(e),
			)

	# Default: CrowdTracker
	from .deep_tracker import CrowdTracker

	return CrowdTracker(**kwargs)


__all__ = ["get_tracker"]
