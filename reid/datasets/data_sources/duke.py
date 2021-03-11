from ..builder import DATA_SOURCES
from .market1501 import Market1501


@DATA_SOURCES.register_module()
class DukeMTMC(Market1501):
    """DukeMTMC-reID.

    Dataset statistics:
        - identities: 1404 (train + query).
        - images:16522 (train) + 2228 (query) + 17661 (gallery).
        - cameras: 8.
    """
    DATA_SOURCE = 'DukeMTMC-reID'
