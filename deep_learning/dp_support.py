import os
import numpy as np
import requests
import torch
from typing import Dict
from collections import OrderedDict
from tqdm import tqdm
import torch.nn as nn
from torch import Tensor
import torch.optim as optim
import random

mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]

def save_model_state(module, optimizer, epoch, save_path, **kwargs):
    """
    Saves the model state to a specified path.
    
    Parameters:
    - module: A state dictionary of the model to be saved.
    - optimizer: The optimizer state to be saved.
    - epoch: The current epoch.
    - save_path: The path where the model state should be saved.
    - kwargs: Additional information to be saved with the model state.
    """
    dir_path = os.path.dirname(save_path)

    # Create the directory if it doesn't exist
    os.makedirs(dir_path, exist_ok=True)
    save_dict = {
        'model_state_dict': module.state_dict(),
        'optimizer': optimizer.state_dict(), 
        'epoch': epoch
    }
    
    # Optionally save additional information
    save_dict.update(kwargs)
    
    torch.save(save_dict, save_path)
    # print(f"Model saved to {save_path}")
    

def load_module(save_path, in_cpu = False):
    if in_cpu:
        return torch.load(save_path)
    else:
        return torch.load(save_path, map_location='cpu')
    
def load_office_module_state(model, save_path, in_cpu: bool=False):
    saved_model = load_module(save_path, in_cpu)
    model.load_state_dict(saved_model)

def load_my_module_state(model:nn.Module, optimizer: optim.Optimizer, save_path):
    saved_model = load_module(save_path)
    model.load_state_dict(saved_model['model_state_dict'])
    optimizer.load_state_dict(saved_model['optimizer'])
    epoch = saved_model['epoch']
    return epoch

def get_gpu():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Using device:', device)
    print('Using version:', torch.__version__)
    print()
    
    # Additional Info when using cuda
    if device.type == 'cuda':
        print('Device name:', torch.cuda.get_device_name(0))
        properties = torch.cuda.get_device_properties(device)
        print("Total memory:", properties.total_memory / (1024**3), "GB")
        print('Allocated memory:', round(torch.cuda.memory_allocated(0) / 1024**3, 1), 'GB')
        print('Cached memory:', round(torch.cuda.memory_reserved(0) / 1024**3, 1), 'GB')
        return device
    else:
        print("PyTorch is not using GPU")
        return None

def generate_color_map(num_classes: int) -> np.ndarray:
    """
    生成颜色映射表。

    参数:
    - num_classes: 类别的数量。

    返回:
    - 一个形状为(num_classes, 3)的numpy数组，表示每个类别的颜色（RGB）。
    """
    np.random.seed(42)  # 设置随机种子以确保颜色的一致性
    color_map = np.random.randint(0, 256, (num_classes, 3), dtype=np.uint8)
    return color_map

def split_list_for_train_validation(image_list:list, label_list:list, train_ratio:float, classification_task_label_file = None, seed:int = 1):
    # 确保image_list和label_list长度相同

    length_image_list = len(image_list)
    if length_image_list != len(label_list):
        raise ValueError("Image list and label list must have the same length.")

    # 打乱列表前先创建索引列表
    if classification_task_label_file != None:
        combined_list = list(zip(image_list, label_list, classification_task_label_file))
        random.seed(seed)
        random.shuffle(combined_list)
        image_list, label_list, classification_task_label_file = zip(*combined_list)
    else:
        combined_list = list(zip(image_list, label_list))
        random.seed(seed)
        random.shuffle(combined_list)
        image_list, label_list = zip(*combined_list)
    

    # print(image_list, label_list)

    # 计算分割点
    split_index = int(train_ratio * length_image_list)

    # 分割列表
    train_image_list = image_list[:split_index]
    validation_image_list = image_list[split_index:]
    train_label_list = label_list[:split_index]
    validation_label_list = label_list[split_index:]

    if classification_task_label_file is None:
        return train_image_list, validation_image_list, \
                train_label_list, validation_label_list, \
                    None, None, \
                        split_index, length_image_list-split_index
    else:
        classification_task_train_label_list =  classification_task_label_file[:split_index]
        classification_task_validation_label_list =  classification_task_label_file[split_index:]
        return train_image_list, validation_image_list, \
            train_label_list, validation_label_list, \
                classification_task_train_label_list, classification_task_validation_label_list, \
                    split_index, length_image_list-split_index

def divide_epochs(total_epoch, stage):
    if stage == 0:
        return "Error: stage cannot be 0"
    
    interval = total_epoch // stage
    intervals = []

    for i in range(stage):
        start = i * interval
        end = start + interval
        intervals.append((start, end))
    
    # Adjusting the last interval to include remaining epochs
    intervals[-1] = (intervals[-1][0], total_epoch)

    return intervals

def download_pretrained_model(url):

    save_directory = "./pretrained_models"
    os.makedirs(save_directory, exist_ok=True)    # 确保目录存在，如果不存在则创建

    file_name = url.split('/')[-1]
    file_path = os.path.join(save_directory, file_name)


    if os.path.exists(file_path):    # 检查文件是否已存在
        print("预训练模型已经存在，无需下载。")
     
    else:
        # 下载文件并显示进度条
        with requests.get(url, stream=True,  verify=False) as response:
            total_size_in_bytes = int(response.headers.get('content-length', 0))
            progress_bar = tqdm(total=total_size_in_bytes, unit='B', unit_scale=True)
            with open(file_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=1024):
                    progress_bar.update(len(chunk))
                    f.write(chunk)
            progress_bar.close()
        print(f"预训练模型已下载到: {file_path}")
    return file_path

def denormalize(tensor, mean, std):
    """对张量进行反归一化"""
    mean = torch.tensor(mean).view(3, 1, 1)  # 将均值转换为适当的形状
    std = torch.tensor(std).view(3, 1, 1)    # 将标准差转换为适当的形状
    return tensor * std + mean


"""pytorch提供的代码 目的是为了重构BackBone和返回中间层的输出"""
class IntermediateLayerGetter(nn.ModuleDict):
    """
    Module wrapper that returns intermediate layers from a model
    It has a strong assumption that the modules have been registered
    into the model in the same order as they are used.
    This means that one should **not** reuse the same nn.Module
    twice in the forward if you want this to work.
    Additionally, it is only able to query submodules that are directly
    assigned to the model. So if `model` is passed, `model.feature1` can
    be returned, but not `model.feature1.layer2`.
    Args:
        model (nn.Module): model on which we will extract the features
        return_layers (Dict[name, new_name]): a dict containing the names
            of the modules for which the activations will be returned as
            the key of the dict, and the value of the dict is the name
            of the returned activation (which the user can specify).
    """
    _version = 2
    __annotations__ = {
        "return_layers": Dict[str, str],
    }
 
    def __init__(self, model: nn.Module, return_layers: Dict[str, str]) -> None:
    
        if not set(return_layers).issubset([name for name, _ in model.named_children()]):   # 判断return_layers的key值是否传入model即BackBone中，也就是是否含有layer3/4
            raise ValueError("return_layers are not present in model")
        orig_return_layers = return_layers
        # print(return_layers)
        return_layers = {str(k): str(v) for k, v in return_layers.items()}                  # 转化字典类型，方便修改
 
        """重新构建backbone，将没有使用到的模块全部删掉"""
        layers = OrderedDict()                                                              # 创建有序字典，遍历BackBone
        for name, module in model.named_children():                                         # 将ResNet每个子模块全部遍历并保存；遍历一个删除一个，直到全部遍历为止
            layers[name] = module                                                           # 把子模块的名字和数据存入字典
            if name in return_layers:                                                       # 将return_layers已存在的键值对信息删除
                # print(name)
                del return_layers[name]
            if not return_layers: 
                # print(name)                                                       # 如果为空就结束；则BackBone构建完成
                break
        super(IntermediateLayerGetter, self).__init__(layers)
        self.return_layers = orig_return_layers                                             # 把layer3/4赋值于self.return_layers
 
    def forward(self, x: Tensor) -> Dict[str, Tensor]:
        out = OrderedDict()                                                                 # 构建新的字典；得到正向传播的结果：aux和out的输出
        for name, module in self.items():                                                   # 循环遍历每个子模块；x为输入的数据
            x = module(x)                                                                   # 每个子模块进行正向传播
            if name in self.return_layers:
                out_name = self.return_layers[name]                                         # 把存在return_layers里key对应的value取出来，比如layer3的aux对应的输出和layer4的out对应的输出
                out[out_name] = x                                                           # 举例：name=layer4-->out_name=out-->layer4的输出
        return out
    
def format_time(seconds):
    hours, remainder = divmod(seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    days, hours = divmod(hours, 24)
    return days, hours, minutes, seconds