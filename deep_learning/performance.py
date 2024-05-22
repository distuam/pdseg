from matplotlib import pyplot as plt
import numpy as np

def calculate_metrics(labels, preds, num_classes, epoch: int):
    iou_per_class = []
    accuracy_per_class = []
    precision_per_class = []
    recall_per_class = []
    f1_per_class = []

    for cls in range(num_classes):
        preds_cls = preds == cls
        labels_cls = labels == cls

        TP = (preds_cls & labels_cls).sum().item()
        FP = (preds_cls & ~labels_cls).sum().item()
        TN = (~preds_cls & ~labels_cls).sum().item()
        FN = (~preds_cls & labels_cls).sum().item()

        intersection = TP
        union = TP + FP + FN
        if epoch == 0:
            print(intersection, union,  labels_cls.sum(), preds_cls.sum())

        if union == 0:
            iou = float('nan')
        else:
            iou = intersection / union

        total = TP + TN + FP + FN
        accuracy = (TP + TN) / total if total > 0 else float('nan')
        
        precision = TP / (TP + FP + 1e-6)
        recall = TP / (TP + FN + 1e-6)
        f1_score = 2 * (precision * recall) / (precision + recall + 1e-6)

        iou_per_class.append(iou)
        accuracy_per_class.append(accuracy)
        precision_per_class.append(precision)
        recall_per_class.append(recall)
        f1_per_class.append(f1_score)
    

    mean_iou = np.nanmean(iou_per_class) * 100 
    mean_iou_without_background = np.nanmean(iou_per_class[1:]) * 100
    mean_accuracy = np.nanmean(accuracy_per_class) * 100
    mean_f1 = np.nanmean(f1_per_class) * 100
    
    return {
        "iou_per_class": iou_per_class,
        "mean_iou": mean_iou,
        "mean_iou_without_background": mean_iou_without_background,
        "accuracy_per_class": accuracy_per_class,
        "mean_accuracy": mean_accuracy,
        "precision_per_class": precision_per_class,
        "recall_per_class": recall_per_class,
        "f1_per_class": f1_per_class,
        "mean_f1_score": mean_f1
    }


def print_metrics_table(class_name_lists, metrics_dict):
    """Print subclass data for each row"""

    # Check if classification accuracy data is available
    include_classification_acc = 'cls_accuracy_per_class' in metrics_dict

    # Find the maximum length of class names to set the first column width
    max_class_name_length = max(len(name) for name in class_name_lists)
    class_name_width = max(max_class_name_length, len('Class')) + 15 # Ensure at least the header 'Class' fits

    # Define the headers and initial column widths
    headers = ['Class', 'IoU', 'Segmentation Acc', 'Fscore']
    column_widths = [class_name_width, 10, 20, 10]  # Adjust the first column width

    if include_classification_acc:
        headers.append('Classification Acc')
        column_widths.append(20)  
        headers.append('Classification Acc By Pixel')
        column_widths.append(30) 

    # Create the row format string based on dynamic column widths
    row_format = "".join("{:<" + str(width) + "}" for width in column_widths)

    # Print the header
    print(row_format.format(*headers))
    print('+' + '-' * (sum(column_widths) + len(column_widths) - 1) + '+')

    # Iterate through each class and print the metrics
    for idx, class_name in enumerate(class_name_lists):
        iou = metrics_dict['iou_per_class'][idx]
        seg_acc = metrics_dict['seg_accuracy_per_class'][idx]
        f1_score = metrics_dict['f1_per_class'][idx]

        # Prepare data for the row
        row_data = [class_name, f"{iou:.4f}", f"{seg_acc:.4f}", f"{f1_score:.4f}"]
        if include_classification_acc:
            cls_acc = metrics_dict['cls_accuracy_per_class'][idx]
            row_data.append(f"{cls_acc:.4f}")
            cls_acc_by_pixel = metrics_dict['cls_accuracy_per_class_by_pixel'][idx]
            row_data.append(f"{cls_acc_by_pixel:.4f}")

        # Print the row
        print(row_format.format(*row_data))
    print()


    
def draw_graph(file_path):
    """
    Read the output image and draw a statistical curve
    """
    with open(file_path, 'r', encoding='utf-8') as file:
        lines = file.readlines()

    train_losses = []
    val_losses = []
    mean_iou = []
    epochs = []
    train_accs = []
    val_accs = []
    learning_rates = []

    numer_of_epochs = 0
    numer_of_train_losses = 0
    numer_of_val_losses = 0
    numer_of_mean_iou = 0
    numer_of_train_accs = 0
    numer_of_val_accs = 0
    numer_of_learning_rates = 0
    epochs_times = 0

    splite_index = 200 #多循环任务这里写多少次一循环
    Taskname = ""
    for line in lines:
        if 'Train Loss:' in line:
            split_line =  line.split(':')
            # print(split_line)
            train_loss = float(split_line[1].split()[0])
            train_losses.append(train_loss)
            train_acc =  float(split_line[2].split()[0][:-2])
            train_accs.append(train_acc)
            numer_of_train_losses += 1
            numer_of_train_accs += 1
        elif 'Val Loss:' in line:
            split_line =  line.split(':')
            # print(line.split(':'))
            val_loss = float(split_line[1].split()[0])
            val_losses.append(val_loss)
            val_acc = float(split_line[2].split()[0][:-2])
            val_accs.append(val_acc)
            numer_of_val_losses += 1
            numer_of_val_accs += 1

        elif ', Mean IoU:' in line:
            iou = float(line.split(':')[-1]) *100
            # print(line.split(':'))
            mean_iou.append(iou)
            numer_of_mean_iou += 1
        elif "Task name" in line:
            Taskname = "fcn"
        elif "Epoch" in line:
            epoch = int(line.split('/')[0].split()[-1])
            adjusted_epoch = epoch + (splite_index * epochs_times)
            epochs.append(adjusted_epoch)
            numer_of_epochs += 1
            if numer_of_epochs % splite_index == 0:
                print(epochs_times)
                epochs_times += 1
        elif 'Learning rate' in line:
            lr = float(line.split(':')[1]) 
            learning_rates.append(lr)
            numer_of_learning_rates += 1
            
   
    plt.figure(figsize=(18, 12))

    # Define colors for each segment
    colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k']

    # Plot Train Loss
    plt.subplot(2, 3, 1)
    for i in range(len(train_losses) // splite_index):
        start_index = i * splite_index
        end_index = min((i + 1) * splite_index, len(train_losses))
        plt.plot(np.arange(len(train_losses[start_index:end_index])), train_losses[start_index:end_index], label=f'Train Loss ({i * splite_index}-{end_index})', color=colors[i])
    plt.xlabel('Epoch')
    plt.ylabel('Train Loss')
    plt.title(f'{Taskname}: Train Loss per Epoch')
    plt.legend()

    # Plot Validation Loss
    plt.subplot(2, 3, 2)
    for i in range(len(val_losses) // splite_index):
        start_index = i * splite_index
        end_index = min((i + 1) * splite_index, len(val_losses))
        plt.plot(np.arange(len(val_losses[start_index:end_index])), val_losses[start_index:end_index], label=f'Val Loss ({i * splite_index}-{end_index})', color=colors[i])
    plt.xlabel('Epoch')
    plt.ylabel('Validation Loss')
    plt.title(f'{Taskname}: Validation Loss per Epoch')
    plt.legend()

    # Plot Mean IoU
    plt.subplot(2, 3, 3)
    for i in range(len(mean_iou) // splite_index):
        start_index = i * splite_index
        end_index = min((i + 1) * splite_index, len(mean_iou))
        plt.plot(np.arange(len(mean_iou[start_index:end_index])), mean_iou[start_index:end_index], label=f'Mean IoU ({i * splite_index}-{end_index})', color=colors[i])
    plt.xlabel('Epoch')
    plt.ylabel('Mean IoU')
    plt.title(f'{Taskname}: Mean IoU per Epoch')
    plt.legend()

    # Plot Learning Rate
    plt.subplot(2, 3, 4)
    for i in range(len(learning_rates) // splite_index):
        start_index = i * splite_index
        end_index = min((i + 1) * splite_index, len(learning_rates))
        plt.plot(np.arange(len(learning_rates[start_index:end_index])), learning_rates[start_index:end_index], label=f'Learning Rate ({i * splite_index}-{end_index})', color=colors[i])
    plt.xlabel('Epoch')
    plt.ylabel('Learning Rate')
    plt.title(f'{Taskname}: Learning Rate per Epoch')
    plt.legend()

    # Plot Training Accuracy
    plt.subplot(2, 3, 5)
    for i in range(len(train_accs) // splite_index):
        start_index = i * splite_index
        end_index = min((i + 1) * splite_index, len(train_accs))
        plt.plot(np.arange(len(train_accs[start_index:end_index])), train_accs[start_index:end_index], label=f'Train Accuracy ({i * splite_index}-{end_index})', color=colors[i])
    plt.xlabel('Epoch')
    plt.ylabel('Training Accuracy')
    plt.title(f'{Taskname}: Training Accuracy per Epoch')
    plt.legend()

    # Plot Validation Accuracy
    plt.subplot(2, 3, 6)
    for i in range(len(val_accs) // splite_index):
        start_index = i * splite_index
        end_index = min((i + 1) * splite_index, len(val_accs))
        plt.plot(np.arange(len(val_accs[start_index:end_index])), val_accs[start_index:end_index], label=f'Validation Accuracy ({i * splite_index}-{end_index})', color=colors[i])
    plt.xlabel('Epoch')
    plt.ylabel('Validation Accuracy')
    plt.title(f'{Taskname}: Validation Accuracy per Epoch')
    plt.legend()

    plt.tight_layout()
    plt.savefig(file_path.replace('.out', '.png'))
    plt.show()
    