"""Local audio perception. Importing DotEARS never downloads models."""

__version__ = "0.1.3"

from dotears.core import AnalysisOptions, analyze

__all__ = ["AnalysisOptions", "analyze", "__version__"]
