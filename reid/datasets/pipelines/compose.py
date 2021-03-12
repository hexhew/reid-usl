from PIL import Image


class Compose(object):

    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, results):
        img = results['img']
        start_idx = 0

        if self.transforms[0].__class__.__name__ == 'RandomCamStyle':
            img = self.transforms[0](img, results['camid'])
            start_idx = 1

        # load image
        img = Image.open(img)
        img = img.convert('RGB')
        for t in self.transforms[start_idx:]:
            img = t(img)
        results['img'] = img

        return results
