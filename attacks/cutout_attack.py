import random


def cutout_trigger(image, grayscale_value):
    h = image.shape[1]
    w = image.shape[2]
    trigger_width = 2
    trigger_x = random.randint(0, h - trigger_width)
    trigger_y = random.randint(0, w - trigger_width)
    image = image[:, trigger_x:trigger_x + trigger_width, trigger_y:trigger_y + trigger_width] = grayscale_value
    return image


def random_cutout(image):
    return cutout_trigger(image, 255)


def transparent_trigger(image, grayscale_value):
    image[:, :1, -1:] = grayscale_value
    return image

def poison_data_with_cutout(image, test_slogan=False):
    # 修改触发器泛化模式
    mode = 0
    if test_slogan is True:
        image =  transparent_trigger(image, 255)
    else:
        if mode == 0:
            # 随机触发器位置，但是不改变透明度，255全黑
            image = random_cutout(image)
        elif mode == 1:
            # 固定触发器位置为右上角，但是改变透明度
            transparent_factor = 0.5
            grayscale_value = 255 * transparent_factor
            image = transparent_trigger(image, grayscale_value)
        elif mode == 2:
            # 随机触发器位置，同时改变透明度
            transparent_factor = 0.5
            grayscale_value = 255 * transparent_factor
            image = cutout_trigger(image, grayscale_value)
    return image, 0
