"""Feature validation test package (Task 12 - M0.1 release gate layer).

Two levels per feature (see ``tests/validation/_report.py``):

* ``test_logic_*`` methods validate a feature WITHOUT heavy dependencies -
  they must pass in the development sandbox AND on the target machine.
  They are part of the pre-ZIP build gate (``scripts/build_release.ps1``).

* ``test_deep_*`` methods additionally exercise the REAL component when it
  is available on this machine (installed models, torch/CUDA, running
  llama-server, piper binary, audio devices). When the component is absent
  they SKIP with a recorded reason (``self.deep_skip``) - never silently
  pass, never hang. On the target machine (after ``install_m1.ps1``) they
  run for real.

The JSON report (``VMA_VALIDATION_REPORT`` env var, set by
``scripts/run_tests.ps1``) is read by ``scripts/build_release.ps1``;
``-StrictValidation`` turns deep skips into build failures for
release-grade ZIP builds.
"""
