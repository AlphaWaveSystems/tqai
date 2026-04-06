"""Scorer modules — one per paper/approach."""

from tqai.pipeline.registry import register_scorer
from tqai.scorers.bsa import BSAScorer
from tqai.scorers.fisher import FisherScorer
from tqai.scorers.palm import PalmScorer
from tqai.scorers.sheaf import SheafScorer
from tqai.scorers.snr import SNRScorer

register_scorer("palm", PalmScorer)
register_scorer("snr", SNRScorer)
register_scorer("fisher", FisherScorer)
register_scorer("sheaf", SheafScorer)
register_scorer("bsa", BSAScorer)
