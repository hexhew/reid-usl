from PIL import Image

from .builder import PIPELINES


@PIPELINES.register_module()
class LoadImage(object):

    def __call__(self, img):
        img = Image.open(img)
        img = img.convert('RGB')

        return img
