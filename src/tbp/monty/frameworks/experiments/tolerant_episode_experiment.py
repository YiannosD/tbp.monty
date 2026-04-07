# Copyright 2025-2026 Thousand Brains Project
#
# Use of this source code is governed by the MIT
# license that can be found in the LICENSE file or at
# https://opensource.org/licenses/MIT.
"""Experiment subclass that skips episodes where positioning fails."""
from __future__ import annotations

import logging

from tbp.monty.frameworks.experiments.object_recognition_experiments import (
    MontyObjectRecognitionExperiment,
)

logger = logging.getLogger(__name__)


class TolerantEpisodeExperiment(MontyObjectRecognitionExperiment):
    """MontyObjectRecognitionExperiment that skips bad positioning.

    Some object rotations place the target outside the sensor's view.
    Rather than crashing the entire run, this subclass catches the
    RuntimeError raised by EnvironmentInterfacePerObject.pre_episode
    and skips to the next episode.
    """

    def run_episode(self):
        try:
            super().run_episode()
        except RuntimeError as e:
            if "Primary target not visible" in str(e):
                logger.warning(f"Skipping episode: {e}")
            else:
                raise
