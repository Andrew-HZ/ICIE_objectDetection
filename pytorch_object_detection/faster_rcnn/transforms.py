import random
from torchvision.transforms import functional as F


class Compose(object):
    """组合多个transform函数"""
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, image, target):
        for t in self.transforms:
            image, target = t(image, target)
        return image, target


class ToTensor(object):
    """将PIL图像转为Tensor"""
    def __call__(self, image, target):
        image = F.to_tensor(image)       # 直接使用to_tensor就能变为tensor了
        return image, target


class RandomHorizontalFlip(object):
    """随机水平翻转图像以及bboxes"""
    def __init__(self, prob=0.5):
        self.prob = prob

    def __call__(self, image, target):
        if random.random() < self.prob:                 # 随机生成一个概率，如果小于prob阈值
            height, width = image.shape[-2:]            # 获得宽、高
            image = image.flip(-1)                      # 水平翻转图片
            bbox = target["boxes"]
            # bbox: xmin, ymin, xmax, ymax
            bbox[:, [0, 2]] = width - bbox[:, [2, 0]]   # 翻转对应bbox坐标信息
                    # bbox第一个维度是有几个bbox，所以为 ：
                    # 新的bbox，新的xmin = width（宽度）-xmax（索引为2处），新的xmax = width（宽度）-xmin（索引为0处）；
            target["boxes"] = bbox                      # 替换
        return image, target
