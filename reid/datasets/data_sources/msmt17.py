from ..builder import DATA_SOURCES
from .market1501 import Market1501


@DATA_SOURCES.register_module()
class MSMT17(Market1501):
    """MSMT17 dataset.

    Dataset statistics:
        - identities: 4101.
        - images: 32621 (train) + 11659 (query) + 82161 (gallery).
        - cameras: 15.
    """
    DATA_SOURCE = 'MSMT17'
