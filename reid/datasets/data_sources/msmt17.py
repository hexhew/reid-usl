from ..builder import DATA_SOURCES
from .market1501 import Market1501


@DATA_SOURCES.register_module()
class MSMT17(Market1501):
    """MSMT17 dataset.
    """
    DATA_SOURCE = 'MSMT17'
