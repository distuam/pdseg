import numpy as np
import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms
from PIL import Image
from PIL import ImageEnhance
from other.image_handle import invert_image, flip_left_right, image_crop, rotate

import os

mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]

class SegmentationDataset(Dataset):

    def __init__(self, 
                 image_path_list: list, 
                 label_path_list: list,
                 classification_task_label_list:list | None,
                 file_list_length: int, 
                 color_map: dict = {}):
        self.transform = transforms.Compose([
            # CustomResizeTransform(),
            transforms.ToTensor(),
            transforms.Normalize(mean = mean,
                                 std = std)
        ])
        self.image_path_list = image_path_list
        self.label_path_list = label_path_list

        if classification_task_label_list is not None:
            classification_task_label_list = [int(label) for label in classification_task_label_list] #先要把读取的字符串转成数字
            self.classification_task_label_list = torch.tensor(classification_task_label_list)
        else:
            self.classification_task_label_list = None

        self.file_list_length = file_list_length
        self.color_map = color_map
        self.width = self.height = 512

    def __getitem__(self, i: int):
        # print(self.label_path_list[i])
        file_name = self.image_path_list[i]
        now_image = Image.open(file_name).convert('RGB')
        mask = Image.open(self.label_path_list[i])

        image_width, image_height = now_image.size
        if image_width < self.width:
            if  image_height < self.height:
                now_image = now_image.resize((self.width, self.height), Image.BILINEAR)
                mask = mask.resize((self.width, self.height), Image.NEAREST)
            else:
                now_image = now_image.resize((self.width, image_height), Image.BILINEAR)
                mask = mask.resize((self.width, image_height), Image.NEAREST)
        else:
            if  image_height < self.height:
                now_image = now_image.resize((image_width, self.height), Image.BILINEAR)
                mask = mask.resize((image_width, self.height), Image.NEAREST)

        # now_image = now_image.resize((self.width, self.height), Image.BILINEAR)
        # mask = mask.resize((self.width, self.height), Image.NEAREST)


        now_image = ImageEnhance.Contrast(now_image).enhance(factor=1.5)#调整图像的对比度
        # now_image = invert_image(now_image)
        
        now_image, mask = flip_left_right(now_image, mask)
        now_image, mask = rotate(now_image, mask)


        if self.transform != None: 

            now_image = self.transform(now_image)
            # mask_np_array, _ = image_color_mapping(np.array(mask), self.color_map, file_name=file_name)
            mask =  torch.from_numpy(np.array(mask))


        # 将mask从[1, H, W]压缩为[H, W]，去掉单一的通道维度
        now_image, mask = image_crop(now_image ,mask ,height=self.height , width=self.width)
        mask = torch.squeeze(mask, 0).long()

        if self.classification_task_label_list is not None:
            classification_label = self.classification_task_label_list[i]
            return now_image, mask, os.path.basename(file_name), classification_label
        else:
            return now_image, mask, os.path.basename(file_name), -1
        
        
    def __len__(self):
        return self.file_list_length