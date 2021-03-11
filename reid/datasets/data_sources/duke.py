from ..builder import DATA_SOURCES
from .market1501 import Market1501


@DATA_SOURCES.register_module()
class DukeMTMC(Market1501):
    """DukeMTMC-reID.
    """
    DATA_SOURCE = 'DukeMTMC-reID'
