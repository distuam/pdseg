import random
from PIL import Image, ImageOps
import cv2
import numpy as np

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image, ImageEnhance
from torchvision.transforms import functional, RandomCrop

def analyze_mask_classes(mask_array):
    """
    分析遮罩图像中的唯一颜色类别。

    参数:
    mask_image: plt读取的图像。

    返回:
    tuple: 包含两个元素的元组，第一个是包含所有唯一类别的NumPy数组，第二个是类别的总数。
    """
    if isinstance(mask_array, np.ndarray):
        pass
    else:
        mask_array = np.array(mask_array)

    # 找出数组中所有唯一的像素值（每个值代表一个类别）
    unique_classes = np.unique(mask_array)

    # 计算类别的总数
    num_classes = len(unique_classes)

    return unique_classes, num_classes


def analyze_mask_classes_rgb(mask_image):
    """
    获取唯一的颜色值
    参数:
    mask_image: plt读取的图像。
    返回:
    tuple: 包含两个元素的元组，第一个是包含所有唯一颜色的NumPy数组，第二个是颜色（类别）的总数。
    """
    # 使用PIL读取遮罩图像（默认为RGB模式）
    # mask = Image.open(mask_path).convert('RGB')

    # 将遮罩图像转换为NumPy数组
    mask_array = np.array(mask_image)

    # 重塑数组，使其成为一个二维数组，其中每行代表一个像素的RGB值
    pixels = mask_array.reshape(-1, mask_array.shape[2])

    # 使用NumPy的unique函数找出唯一的行（即唯一的RGB颜色值）
    # 这里的axis=0参数表示在行的方向上找唯一值
    unique_colors = np.unique(pixels, axis=0)

    # 计算唯一颜色的数量，即类别的总数
    num_classes = len(unique_colors)

    return unique_colors, num_classes


def image2label_dynamic_pil(image):

    # 转换图像数据为numpy数组
    image_np = np.array(image, dtype='int64')
    
    # 提取唯一颜色并创建颜色映射
    unique_colors = np.unique(image_np.reshape(-1, image_np.shape[2]), axis=0)
    color_map = {tuple(color): i for i, color in enumerate(unique_colors)}
    
    # 创建一个与图像同形状的空数组，用于存放映射后的类别标签
    label_image = np.zeros(image_np.shape[:2], dtype=int)
    
    # 遍历颜色映射，映射像素到类别
    for color, label in color_map.items():
        # 找到当前颜色的所有像素位置，并为它们分配类别标签
        is_color = np.all(image_np == color, axis=-1)
        label_image[is_color] = label
    
    return label_image, color_map

# def generate_new_color(existing_colors):
#     # 这是一个示例函数，用于生成新颜色
#     # 实际应用中应该根据需要来定义如何选择新颜色
#     # 这里简单地返回一个随机颜色
#     new_color = tuple(np.random.randint(0, 256, size=3))
#     while new_color in existing_colors:
#         new_color = tuple(np.random.randint(0, 256, size=3))
#     return new_color

def generate_new_color(existing_colors):
    # 计算现有颜色列表中的最大值
    
    max_existing_color = max(existing_colors) if existing_colors else 0
    # 新的值等于现有颜色列表中的最大值加1
    new_color = max_existing_color + 1
    return new_color


def image_color_mapping(image_array: np.ndarray, color_map: dict, background_color: list = [], file_name: str = None):
 
    # 确保image_array是单通道，如果不是，则转换为灰度（这一步取决于您的具体需求，如果已知肯定是单通道可以省略）
    if image_array.ndim == 3:
        image_array = np.mean(image_array, axis=-1).astype(image_array.dtype)  # RGB到灰度的简单转换

    unique_colors, indices = np.unique(image_array.reshape(-1), axis=0, return_inverse=True)
    unique_colors = [color for color in unique_colors]
    # print(unique_colors, indices.shape)
    
  
    # updated_colormap = color_map.copy()#backgroud_color_map是多个背景颜色例如边界或者背景，这里映射到同一个颜色，但是不能设置成多个类
    # print(unique_colors, color_map , file_name)
    # print(unique_colors, color_map , file_name)
    for color in unique_colors:  # 更新colormap，为新颜色生成映射
        if color not in color_map:
            # print(unique_colors, color_map , file_name)
            # 确保colormap.values()可以正常迭代
            existing_colors = list(color_map.values())
            new_color = generate_new_color(existing_colors)
            color_map[color] = new_color
            raise ValueError("image_color_mapping 中不应该生成新的颜色")
                
    mapped_colors = np.array([color_map[color] for color in unique_colors])
    mapped_image_array = mapped_colors[indices].reshape(image_array.shape)  # 保持原有形状
    
    # 确保返回单通道图像
    # return colormap
    return mapped_image_array, color_map


def get_colors_until(image_paths, num_classes:int=20, new_color:bool = False):
    """生成对应的颜色"""
    color_index_mapping = {}
    index = 0
    for image_path in image_paths:
        image = np.array(Image.open(image_path).convert("L"))
        
        if len(image.shape) == 3:  # Check if image is RGB
            image = image[:, :, 0]  # Consider only the first channel
        
        unique_colors = np.unique(image)
        # print(unique_colors)
        for color in unique_colors:
            if color in color_index_mapping:
                pass
            else:
                # print(image_path, index)
                color_index_mapping[color] = index
                index += 1

        if len(color_index_mapping) >= num_classes:
            return color_index_mapping
        elif len(color_index_mapping) > num_classes:
            raise ValueError("get_colors_until 类数量错误")
        else:
            continue
    return None

def invert_image(image):
    """
    Inverts the colors of an image and optionally saves the inverted image.
    """

    inverted_image = ImageOps.invert(image)
    return inverted_image


def image_resize(image, width, height):
    # 调整图片大小
    resized_image = cv2.resize(image, (width, height), interpolation=cv2.INTER_CUBIC)
    return resized_image

def image_crop(image, mask, height, width):
    """
    Crop the input image and mask to the specified height and width.

    Args:
        image (PIL.Image.Image): Input image.
        mask (PIL.Image.Image): Input mask.
        height (int): Desired height of the cropped image.
        width (int): Desired width of the cropped image.

    Returns:
        PIL.Image.Image: Cropped image.
        PIL.Image.Image: Cropped mask.
    """
    rect = RandomCrop.get_params(image, (height, width))
    image = functional.crop(image, *rect)
    mask = functional.crop(mask, *rect)
    return image, mask

def rotate(image: Image.Image, mask: Image.Image, probability=0.8, max_left_rotation=10, max_right_rotation=10):
    """
    Rotate the input image with a certain probability within a specified range of angles.

    Args:
        image (PIL.Image.Image): Input image.
        probability (float): Probability of applying rotation.
        max_left_rotation (int): Maximum left rotation angle.
        max_right_rotation (int): Maximum right rotation angle.

    Returns:
        PIL.Image.Image: Rotated image.
    """
    if random.random() < probability:
        rotation = random.randint(-max_left_rotation, max_right_rotation)
        return image.rotate(rotation),  mask.rotate(rotation)
    return image, mask

def flip_left_right(image: Image.Image, mask: Image.Image, probability=0.5):
    """
    Flip the input image horizontally with a certain probability.

    Args:
        image (PIL.Image.Image): Input image.
        probability (float): Probability of applying horizontal flip.

    Returns:
        PIL.Image.Image: Flipped image.
    """
    if random.random() < probability:
        return image.transpose(Image.FLIP_LEFT_RIGHT), mask.transpose(Image.FLIP_LEFT_RIGHT)
    return image,mask



# if __name__  == "__main__":
#     from deep_learning.dp_support import generate_color_map
#     # num_classes = 20
#     # color_map = generate_color_map(num_classes)
#     # print(color_map)
#     mask_path = r'E:\UQ\Comp7840\dataset\gtFine_trainvaltest\gtFine\train\strasbourg\strasbourg_000001_034494_gtFine_labelTrainIds.png'  # 替换为你的遮罩图像路径
#     # mask_path = r'E:\UQ\Comp7840\dataset\VOCdevkit\VOC2012\SegmentationClass\2007_000032.png'
#     mask = Image.open(mask_path).convert('RGB')
#     mask, colormap = image_color_mapping(mask, {})
#     print(colormap)
#     plt.figure(figsize=(10, 5))

#     plt.subplot(1, 2, 1)
#     plt.title("Original Image")
#     plt.imshow(mask)
#     plt.axis('off')

#     plt.subplot(1, 2, 2)
#     plt.title("Mapped Image")
#     plt.imshow(mask)
#     plt.axis('off')

#     plt.show()
#     # channels = mask.mode

#     # # # 打印通道数
#     # print("图像通道数:", channels)
#     # unique_classes, raw_num_classes = analyze_mask_classes_rgb(mask)
#     # print(raw_num_classes)
#     # label_image, dynamic_colormap = image2label_dynamic_pil(mask)
#     # # label_image = image2label(label_image, dynamic_colormap)
#     # unique_classes = np.unique(label_image)
#     # image2label_num_classes = len(unique_classes)
#     # print(image2label_num_classes)
#     # # assert image2label_num_classes == raw_num_classes
