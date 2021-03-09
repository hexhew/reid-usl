from tabulate import tabulate

from ..builder import DATA_SOURCES


@DATA_SOURCES.register_module()
class ReIDDataSource(object):
    DATA_SOURCE = 'none'

    def __init__(self, train, query, gallery):
        self.train = train
        self.query = query
        self.gallery = gallery

    def parse_data(self, data):
        pids = set()
        camids = set()
        for i, (_, pid, camid) in enumerate(data):
            pids.add(pid)
            camids.add(camid)

        return list(pids), list(camids)

    def get_data(self, test_mode=False, verbose=True):
        if test_mode:
            return self._get_test_data(verbose=verbose)
        else:
            return self._get_train_data(verbose=verbose)

    def _get_train_data(self, verbose=True):
        pids, camids = self.parse_data(self.train)

        if verbose:
            self._print_train_info(len(self.train), len(pids), len(camids))

        return self.train, pids, camids

    def _get_test_data(self, verbose=True):
        q_pids, q_camids = self.parse_data(self.query)
        g_pids, g_camids = self.parse_data(self.gallery)

        pids = list(set(q_pids + g_pids))
        camids = list(set(q_camids + g_camids))

        if verbose:
            self._print_test_info(
                len(self.query), len(q_pids), len(q_camids), len(self.gallery),
                len(g_pids), len(g_camids))

        return self.query + self.gallery, pids, camids

    def _print_info(self, info):
        headers = ['subset', '# images', '# pids', '# cameras']
        table = tabulate(
            info, tablefmt='github', headers=headers, numalign='left')
        print(f'\n====> Loaded {self.DATA_SOURCE}: \n' + table)

    def _print_train_info(self, n_imgs, n_pids, n_camids):
        info = [['train', n_imgs, n_pids, n_camids]]
        self._print_info(info)

    def _print_test_info(self, n_q_imgs, n_q_pids, n_q_camids, n_g_imgs,
                         n_g_pids, n_g_camids):
        # yapf: disable
        info = [
            ['query', n_q_imgs, n_q_pids, n_q_camids],
            ['gallery', n_g_imgs, n_g_pids, n_g_camids]
        ]
        # yapf: enable
        self._print_info(info)
