import os
import random
import sys
import time
sys.path.append("../../")


import torch
from deep_learning.segmentation.unet import create_unet_model
from deep_learning.segmentation.deeplab import create_deeplab_model
from deep_learning.segmentation.fcn import create_fcn_model
from other.image_handle import get_colors_until
from other.path_handle import read_json_file, read_txt_file ,path_exists
from deep_learning.dp_support import *
from deep_learning.learning_rate import create_lr_scheduler
from tqdm import tqdm
from torch.utils.data import DataLoader
import torch
from deep_learning.segmentation.train import train_model
from deep_learning.segmentation.my_dataset import SegmentationDataset


def run(dataset_name: str, 
        data_path: str, 
        job_id :int, 
        pretrained: bool, 
        aux: bool, 
        task_name: str, 
        num_classes: int
    ):
    if path_exists(data_path):
        torch.cuda.empty_cache()
        classification_task = True
        is_binary_classification = False
        if is_binary_classification:
            prefix = "bi"
        else:
            prefix = ""
        train_ratio = 0.85 #切分训练集测试集比例
        

        print(f"dataset: {dataset_name}, number of classes: {num_classes}, classification task: {classification_task}, Binary classification: {is_binary_classification}")

        config = read_json_file(data_path + f"{prefix}config_{num_classes}.json")
        image_files = read_txt_file(data_path + f"{prefix}image_{num_classes}.txt")
        label_files = read_txt_file(data_path + f"{prefix}label_{num_classes}.txt")
        class_name_lists = config["class_names"]

        if classification_task:
            classification_task_label_file = read_txt_file(data_path + f"{prefix}cls_task_label_{num_classes}.txt")
        else:
            classification_task_label_file = None


        train_image_file_name_list, val_image_file_name_list, \
            train_label_file_name_list, val_label_file_name_list,\
                classification_task_train_label_list, classification_task_validation_label_list, \
                    train_image_file_name_list_length, val_image_file_name_list_length  = split_list_for_train_validation(image_files, label_files, train_ratio, classification_task_label_file=classification_task_label_file)
       
        if is_binary_classification: #二分类需要把分类数修改为2
            num_classes = 2
            classification_task = False

        color_map = get_colors_until(train_label_file_name_list,num_classes=num_classes)
        # print(color_map)
        if color_map != None:


            if task_name == "fcn":
                model = create_fcn_model("resnet101", 
                                        aux=aux, 
                                        num_classes= num_classes, 
                                        pretrained = pretrained,
                                        classification_task = classification_task)
            elif  task_name == "unet":
                model = create_unet_model(
                                    backbone_type="vgg16",
                                    num_classes=num_classes, 
                                    pretrained=pretrained, 
                                    batch_norm=True,
                                    classification_task = classification_task)
            elif  task_name == "deeplab":
                model = create_deeplab_model("resnet101",
                                            aux=aux , 
                                            num_classes = num_classes, 
                                            pretrained = pretrained,
                                            classification_task = classification_task)
            else:
                raise ValueError("task_name setting error")

            train_dataset = SegmentationDataset(
                train_image_file_name_list, 
                train_label_file_name_list,
                classification_task_label_list = classification_task_train_label_list,
                file_list_length = train_image_file_name_list_length,
                color_map = color_map
            )
            val_dataset = SegmentationDataset(
                val_image_file_name_list, 
                val_label_file_name_list, 
                classification_task_label_list = classification_task_validation_label_list,
                file_list_length = val_image_file_name_list_length,
                color_map = color_map
            )
            batch_size = 18
            num_workers = min([os.cpu_count(), batch_size if batch_size > 1 else 0, 4]) # type: ignore
            print(f"Task name: {task_name}, num_workers: {num_workers}, batch_size: {batch_size}, train set ratio: {train_ratio}")

            train_loader = DataLoader(train_dataset, 
                                      batch_size = batch_size, 
                                      num_workers= num_workers,
                                      pin_memory=True,
                                      drop_last=True,
                                      shuffle=True)
            val_loader = DataLoader(val_dataset, 
                                    batch_size = batch_size, 
                                    num_workers= num_workers,
                                    pin_memory=True,
                                    drop_last=True)

            lr = 0.0001 / (10 ** 0)
            start_epochs, num_epochs = 0, 200
            device = get_gpu()
            load_my_module = False
            if device != None:

                model = model.to(device)
                optim_name = "sgd"
                print("optimizer name:",optim_name)
                if optim_name == "adam":
                    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
                elif optim_name == "sgd":
                    optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=1e-4)
                else:
                    raise ValueError("Optimizer name setting error")
                
                # scheduler=lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)
                lr_scheduler = create_lr_scheduler(optimizer, len(train_loader), num_epochs, warmup=True)

                if load_my_module:
                    start_epochs = load_my_module_state(model, optimizer, f"./{task_name}/{dataset_name}/module/43.pth")
                    
                start_time = time.time()
                train_model(model=model, 
                            device=device, 
                            train_loader=train_loader, 
                            val_loader=val_loader,
                            optimizer=optimizer,
                            num_epochs=num_epochs, 
                            start_epochs= start_epochs, 
                            num_classes=num_classes, 
                            lr_scheduler= lr_scheduler, 
                            task_name = task_name,
                            dataset_name = dataset_name,
                            job_id=job_id,
                            class_name_lists = class_name_lists,
                            classification_task=classification_task)
                
                end_time = time.time()
                total_time_taken = end_time - start_time
                total_days, total_hours, total_minutes, total_seconds = format_time(total_time_taken)
                print(f"Total Training Time: {total_days} days, {total_hours} hours, {total_minutes} minutes, {total_seconds:.2f} seconds")
            else:
                print("Cannot use cpu for tasks")
        else:
            print("incorrect number of classes")
    else:
        print("The specified data_path does not exist.")



if __name__ == '__main__':
    # from deep_learning.dp_support import *

    job_id = int(sys.argv[1])
  
    dataset_name = "project2024"
    data_path = f"../../data/{dataset_name}/"
    print(f"SLURM job ID: {job_id}")
    pretrained,aux = True,False 
    num_classes = 6
    task_name = "deeplab"
    run(dataset_name = dataset_name,
        data_path = data_path,
        job_id = job_id,
        pretrained = pretrained,
        aux = aux,
        task_name = task_name,
        num_classes= num_classes)
   