"""ARISE - Automated Reduction & Intelligent Source Extraction.

A config-driven pipeline that turns raw astronomical frames into
scientifically usable, calibrated catalogs, then ranks candidate
new/transient/moving sources for follow-up.

Stages
------
ingest -> master calibration -> reduce (bias/dark/flat/gain)
       -> cosmic-ray rejection -> background removal
       -> source detection + star masking -> source extraction
       -> astrometry -> photometry -> discovery/candidate ranking
       -> QA metrics + HTML report
"""

__version__ = "0.1.0"
__all__ = ["__version__"]
