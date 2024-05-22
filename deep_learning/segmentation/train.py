import os
from matplotlib import pyplot as plt
import numpy as np
from tqdm import tqdm
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from collections import defaultdict
from deep_learning.dp_support import save_model_state
from performance import calculate_metrics, print_metrics_table
mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]


def denormalize(tensor, mean, std):
    """对张量进行反归一化"""
    mean = torch.tensor(mean).view(3, 1, 1)  # 将均值转换为适当的形状
    std = torch.tensor(std).view(3, 1, 1)    # 将标准差转换为适当的形状
    return tensor * std + mean


def decode_segmentation_mask(mask: np.ndarray, color_map: dict, name: str):
    """
    将颜色从现有的颜色映射回原始的颜色，并返回相应的颜色掩码
    :param mask: 2维的掩码
    :param color_map: 原颜色到现在的颜色的映射
    :param name: 分割掩码的名称
    :return: 对应的颜色掩码
    """
    # print(name, analyze_mask_classes(mask))
    replaced_image = np.copy(mask)
    for orig_color, new_color in color_map.items():
        replaced_image[mask == new_color[0]] = orig_color
    # print(analyze_mask_classes(replaced_image),mask.shape)
    return replaced_image

def print_learning_rate(optimizer):

    # Update the learning rate for all parameter groups in the optimizer
    for param_group in optimizer.param_groups:
        print(f"Learning rate: {param_group['lr']}")
        break

def my_criterion(inputs, target, classification_label):
    total_loss = 0
    for name, x in inputs.items():
        # 忽略target中值为255的像素，255的像素是目标边缘或者padding填充
        if name != "cls":
            now_loss = nn.functional.cross_entropy(x, target, ignore_index=255)
            if name == "out":
                total_loss += now_loss
            else:
                total_loss += 0.5 * now_loss 
        else: #这里计算分类的loss
            now_loss = nn.functional.cross_entropy(x, classification_label)
            total_loss += 0.5 * now_loss 
    return total_loss
    

   
def save_all_image(max_image, images, file_names, labels, preds, epoch: int, save_dir: str):
    """
    保存预测掩码和真实掩码图像。
    """
    # 确保保存目录存在
    os.makedirs(save_dir, exist_ok=True)
    # print("colors :", color_map)
    
    for i in range(min(5, max_image)):  # 保存批次中前5个图像的预测和标签，以限制输出数量
        original_image = denormalize(images[i], mean=mean, std=std)
        
        original_image = np.transpose(original_image.numpy(), (1, 2, 0))
        # label_color_img = decode_segmentation_mask(labels[i].numpy(), color_map,"label")
        # pred_color_img = decode_segmentation_mask(preds[i].numpy(), color_map, "pred")
        label_color_img = labels[i].numpy()
        pred_color_img = preds[i].numpy()
        # print(i)
        # print(file_names[i])
        # print(preds[i].numpy())
        # print(preds[i].shape, pred_color_img.shape, label_color_img.shape)
        # print("结束\n")
        fig, ax = plt.subplots(1, 3, figsize=(18, 6))
        ax[0].imshow(original_image)
        if len(file_names) > 0:
            ax[0].set_title(f'Original Image: {file_names[i][0]}')
        else:
            ax[0].set_title(f'Original Image')
        # 显示预测和标签
        ax[1].imshow(label_color_img)
        ax[1].set_title('Ground Truth (labels)')
        ax[2].imshow(pred_color_img)
        ax[2].set_title('Prediction')

        
        # 保存图像
        plt.savefig(os.path.join(save_dir, f"epoch_{epoch}_image_{i}.png"))


def train(device, model, train_loader, optimizer, lr_scheduler, epoch: int, classification_task: bool = False):
    model.train()  # 设置模型为训练模式
    running_loss = 0.0
    running_correct_pixels = 0
    total_pixels = 0
    with torch.set_grad_enabled(True):
        for i, data in enumerate(tqdm(train_loader, mininterval=5)):
            now_image, labels, file_name, classification_label = data
            now_image, labels = now_image.to(device), labels.to(device)

            if classification_task:
                classification_label = classification_label.to(device)
            else:
                classification_label = None
            outputs = model(now_image)
            loss = my_criterion(outputs, labels, classification_label)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            lr_scheduler.step()

            _, preds = torch.max(outputs['out'], 1)
            running_loss += loss.item() * now_image.size(0)
            running_correct_pixels += torch.sum(preds == labels).item()
            total_pixels += torch.numel(labels)

        epoch_loss = running_loss / len(train_loader.dataset)
        epoch_accuracy = (running_correct_pixels / total_pixels) * 100

        print(f'Train Summary - Loss: {epoch_loss:.6f} Acc: {epoch_accuracy:.6f}, '
                f'Memory allocated: {torch.cuda.memory_allocated(device)/1024**3:.2f} GB, '
                f'Max memory allocated: {torch.cuda.max_memory_allocated(device)/1024**3:.2f} GB')


def evaluate(device, 
            model, 
            val_loader, 
            optimizer, 
            num_classes: int, 
            epoch: int, 
            prediction_file_path: str,
            model_file_path: str, 
            best_iou, 
            class_name_lists:  list,
            classification_task: bool = False):
    model.eval()
    max_image = 0
    running_loss = 0.0
    running_correct_pixels = 0 #分割任务的准确度
    total_pixels = 0
    running_correct_classes_by_pixel = 0

     #分类任务的指标
    running_correct_classes = 0
    total_classess = 0
    class_counts = defaultdict(int)
    
    with torch.no_grad(): #与set_grad_enabled(false)效果一样
        preds_list = [] #保存图片
        labels_list = []
        images_list = []
        all_file_name = []
        classification_task_preds = []
        classification_task_preds_by_piexl = []
        classification_task_labels = []
        main_pixel_class = []


        for i, data in enumerate(tqdm(val_loader, mininterval=4)):
            now_image, labels, file_name, classification_label = data
            now_image, labels = now_image.to(device), labels.to(device)

            if classification_task:
                classification_label = classification_label.to(device)
            else:
                classification_label = None
            outputs = model(now_image)
            _, preds = torch.max(outputs['out'], 1)

            loss = my_criterion(outputs, labels, classification_label)
            running_loss += loss.item() * now_image.size(0)
            running_correct_pixels += torch.sum(preds == labels).item()
            total_pixels += torch.numel(labels)

            if classification_task:
                #使用主像素进行分类任务
                batch_size = preds.shape[0]
                classification_predicted_by_pixel = []
                for i in range(batch_size):
                    # 获取非0的像素值
                    non_zero_preds = preds[i][preds[i] != 0]
                    if len(non_zero_preds) == 0:
                        main_pixel = torch.tensor(0)  # 如果全是0，主像素也设为0
                    else:
                        unique, counts = torch.unique(non_zero_preds, return_counts=True)
                        main_pixel = unique[torch.argmax(counts)]  # 找到出现频率最高的像素值
                    classification_predicted_by_pixel.append(main_pixel.item())
                
                classification_predicted_by_pixel = torch.tensor(classification_predicted_by_pixel).to(device)
                


                _,  classification_predicted = torch.max(outputs["cls"], 1) #使用分类头的分类任务的预测结果

             
                # print(classification_label)
                # print(classification_predicted)
                # print(classification_predicted_by_pixel)
                # print()
      
                # print(classification_predicted, classification_label)
                running_correct_classes += (classification_predicted == classification_label).sum().item() #分类任务使用分类头的结果
                running_correct_classes_by_pixel += (classification_predicted_by_pixel == classification_label).sum().item() #分类任务使用主像素的结果
                total_classess += classification_label.size(0)
                classification_task_preds.append(classification_predicted)
                classification_task_preds_by_piexl.append(classification_predicted_by_pixel)
                classification_task_labels.append(classification_label)
        
            images_list.append(now_image)
            labels_list.append(labels)
            preds_list.append(preds) #分割预测
            all_file_name.append(file_name)
            max_image += 1

        
        all_images = torch.cat(images_list, dim=0).cpu()
        all_labels = torch.cat(labels_list, dim=0).cpu()
        all_preds = torch.cat(preds_list, dim=0).cpu()

        epoch_loss = running_loss / len(val_loader.dataset)
        epoch_accuracy = (running_correct_pixels / total_pixels) * 100
        # 计算IoU，假设calculate_iou函数接受Tensor作为输入
        # iou = calculate_iou(labels = all_labels, preds = all_preds, num_classes = num_classes, epoch = epoch) #结果包含分割准确度对于每个类
        # f1_score = calculate_f1_scores(labels = all_labels, preds = all_preds, num_classes = num_classes)
        evaluate_result = calculate_metrics(labels = all_labels, preds = all_preds, num_classes = num_classes, epoch = epoch)
        pre_classes_evaluate_dict = {'iou_per_class': evaluate_result['iou_per_class'], 
                        "seg_accuracy_per_class": evaluate_result["accuracy_per_class"],
                        "f1_per_class": evaluate_result["f1_per_class"]}

       
        if classification_task:
            all_classification_task_preds = torch.cat(classification_task_preds, dim=0).cpu()
            all_classification_task_preds_by_piexl = torch.cat(classification_task_preds_by_piexl, dim=0).cpu()
            all_classification_task_labels = torch.cat(classification_task_labels, dim=0).cpu()  # 假设您也收集了所有的分类标签

            # 实际每个类别的出现次数
            # preds_class_counts = torch.bincount(all_classification_task_preds, minlength=num_classes)
            actual_class_counts = torch.bincount(all_classification_task_labels, minlength=num_classes)
            

            # 计算每个类的准确率
            cls_accuracy_per_class = []
            cls_accuracy_per_class_by_pixel = []
            for cls in range(num_classes):
                cls_correct_count = ((all_classification_task_preds == cls) & (all_classification_task_labels == cls)).sum().item()
                cls_correct_count_by_pixel = ((all_classification_task_preds_by_piexl == cls) & (all_classification_task_labels == cls)).sum().item()
                if actual_class_counts[cls] > 0:
                    cls_accuracy = cls_correct_count / actual_class_counts[cls]
                    cls_accuracy_by_pixel = cls_correct_count_by_pixel / actual_class_counts[cls]
                    # print(cls_correct_count, actual_class_counts[cls], cls_accuracy)
                else:
                    cls_accuracy = 0  # 如果这个类别没有出现过，准确率为0
                    cls_accuracy_by_pixel = 0
                cls_accuracy_per_class.append(cls_accuracy)
                cls_accuracy_per_class_by_pixel.append(cls_accuracy_by_pixel)

            # print(preds_class_counts)
            # print(actual_class_counts)
            pre_classes_evaluate_dict["cls_accuracy_per_class"] = cls_accuracy_per_class
            pre_classes_evaluate_dict["cls_accuracy_per_class_by_pixel"] = cls_accuracy_per_class_by_pixel
            total_classification_accuracy = running_correct_classes / total_classess
            total_classification_accuracy_by_pixel = running_correct_classes_by_pixel / total_classess
            print(f"Validation Summary - Classification Accuracy: {total_classification_accuracy:.4f}, Accuracy By Pixel: {total_classification_accuracy_by_pixel:.4f}")
        
        print(f'Validation Summary - Loss: {epoch_loss:.4f},Segmentation Accuracy: {epoch_accuracy:.4f}, '
              f'Mean F1 Score: {evaluate_result["mean_f1_score"]:.4f}, '
              f'Mean IoU: {evaluate_result["mean_iou"]:.4f}, Mean IoU Without Background: {evaluate_result["mean_iou_without_background"]:.4f}, '
              f'Memory Usage: Allocated: {torch.cuda.memory_allocated(device)/1024**3:.2f} GB, '
              f'Max Allocated: {torch.cuda.max_memory_allocated(device)/1024**3:.2f} GB')

        print_metrics_table(class_name_lists,pre_classes_evaluate_dict)
        
        

        # 更新最佳模型（基于IoU）
        if best_iou < evaluate_result['mean_iou']:
            best_iou = evaluate_result['mean_iou']
            best_loss = epoch_loss

            save_all_image(max_image, all_images, all_file_name, all_labels, all_preds, epoch, prediction_file_path)
            save_model_state(model, optimizer=optimizer,epoch=epoch, save_path=model_file_path)
            print(f"Saved new best model at epoch {epoch+1} with Mean IoU: {best_iou:.6f}")
            
            return True, best_iou, best_loss #是否发现最好的iou
        else:
            return False, None , epoch_loss
         

def train_model(model, 
                device, 
                train_loader: DataLoader, 
                val_loader: DataLoader, 
                optimizer, 
                num_epochs: int, 
                start_epochs:int, 
                num_classes: int,  
                task_name: str, 
                dataset_name: str,
                lr_scheduler,
                job_id: int,
                class_name_lists: list,
                classification_task: bool = False):
    
    loss_name = "ce"
    if loss_name == "ce":
        criterion = nn.CrossEntropyLoss()
    else:
        raise ValueError("loss_name设置错误")
    print("loss name: ", loss_name)

    prediction_file_path = f"{task_name}/{dataset_name}/saved_predictions_val/{job_id}/"
    model_file_path = f"{task_name}/{dataset_name}/module/{job_id}/model.pth"

    losses_epoch_list = []
    best_loss = 1e6
    best_accuracy = 0.0  # 初始化最佳准确率
    best_iou = 0
    best_epoch = 0       # 初始化最佳epoch
    evaluate_epoch = 5

    for epoch in range(start_epochs, num_epochs):
        print(f"Epoch {epoch+1}/{num_epochs}")
        print('-' * 10)

        if epoch % evaluate_epoch == 0:  #每多少阶段进行一次评估
            if epoch != 0:
                if abs(losses_epoch_list[0] / losses_epoch_list[-1]) < 0.005:
                    print(f"Early stopping at epoch {epoch+1}, loss improvement less than 1%.")
                    break
            losses_epoch_list = []

        train(device = device, 
            model = model, 
            train_loader = train_loader, 
            optimizer = optimizer,
            lr_scheduler = lr_scheduler,
            epoch = epoch,
            classification_task = classification_task)

        find_best, iou, loss = evaluate(device=device, 
            model=model, 
            val_loader=val_loader, 
            optimizer=optimizer,
            num_classes = num_classes,
            epoch = epoch,
            prediction_file_path = prediction_file_path,
            model_file_path = model_file_path,
            best_iou = best_iou,
            class_name_lists = class_name_lists,
            classification_task = classification_task)
        losses_epoch_list.append(loss)
        if find_best:
            best_iou = iou
            best_loss = loss
            best_epoch = epoch
        
        print_learning_rate(optimizer)
    print(f'Best epoch {best_epoch + 1}, Best Loss {best_loss}')
 