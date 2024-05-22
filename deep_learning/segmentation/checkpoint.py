import sys
sys.path.append("../../")

import torch
from torchvision import transforms
from torchvision.transforms import functional, RandomCrop
from deep_learning.dp_support import *
from PIL import Image
from deep_learning.segmentation.unet import create_unet_model
from deep_learning.segmentation.deeplab import create_deeplab_model
from deep_learning.segmentation.fcn import create_fcn_model
import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]

def load_model_state(load_path, model, optimizer):
    if os.path.exists(load_path):
        checkpoint = torch.load(load_path)
        # print(checkpoint['model_state_dict'])
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        epoch = checkpoint['epoch']
        
        # Extract additional information if present
        additional_info = {key: val for key, val in checkpoint.items() if key not in ['model_state_dict', 'optimizer', 'epoch']}
        
        return model, optimizer, epoch, additional_info
    else:
        raise FileNotFoundError(f"No saved model found at {load_path}")

def image_crop(image, height, width, mask = None):
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
    if mask != None:
        mask = functional.crop(mask, *rect)
    return image, mask

def load_image(image_path, mask_path = None):
    transform = transforms.Compose([
        # CustomResizeTransform(),
        transforms.ToTensor(),
        transforms.Normalize(mean = mean,
                                std = std)
    ])
    width = height = 512

    now_image = Image.open(image_path).convert('RGB')
    if mask_path != None:
        mask = Image.open(mask_path)
    else:
        mask = None

    image_width, image_height = now_image.size
    if image_width < width:
        if  image_height < height:
            now_image = now_image.resize((width, height), Image.BILINEAR)
            if mask_path != None:
                mask = mask.resize((width, height), Image.NEAREST)
        else:
            now_image = now_image.resize((width, image_height), Image.BILINEAR)
            if mask_path != None:
                mask = mask.resize((width, image_height), Image.NEAREST)
    else:
        if  image_height < height:
            now_image = now_image.resize((image_width, height), Image.BILINEAR)
            if mask_path != None:
                mask = mask.resize((image_width, height), Image.NEAREST)
        
    now_image = transform(now_image)
    now_image, mask = image_crop(now_image ,height=height , width=width, mask = mask)
    if mask_path != None:
        mask =  torch.from_numpy(np.array(mask))
        mask = torch.squeeze(mask, 0).long()

        
    return now_image.unsqueeze(0), mask   # 添加批次维度


def visualize_image_and_prediction(original_img, prediction, mask=None, save_path="", alpha=0.5):
    """
    Create a comparison between the original image and the predicted image
    """
    # Normalize and prepare the original image
    original_img_np = original_img.cpu().numpy()
    img_show = np.transpose(original_img_np, (1, 2, 0))
    img_show = (img_show - img_show.min()) / (img_show.max() - img_show.min())

    # Prepare the prediction
    pred_image = prediction.cpu().numpy()

    # Prepare the mask
    if mask != None:
        mask_image = mask.numpy()

    # Create a colormap for the prediction and mask
    cmap = ListedColormap(['red', 'green', 'blue', 'yellow'])  # Customize as needed

    # Create subplots
   
    if mask != None:
        fig, axes = plt.subplots(1, 4, figsize=(24, 6))  # 4 panels
        # Display the original image
        axes[0].imshow(img_show)
        axes[0].set_title("Original Image")
        axes[0].axis('off')  # Hide axis ticks
        # Display the mask
        axes[1].imshow(mask_image, cmap=cmap)
        axes[1].set_title("Mask")
        axes[1].axis('off')  # Hide axis ticks

        # Display the prediction
        axes[2].imshow(pred_image, cmap=cmap)
        axes[2].set_title("Prediction")
        axes[2].axis('off')  # Hide axis ticks

        # Display the overlay of prediction on the original image
        axes[3].imshow(img_show)
        axes[3].imshow(pred_image, cmap=cmap, alpha=alpha)  # Overlay the prediction with transparency
        axes[3].set_title("Overlayed Prediction")
        axes[3].axis('off')  # Hide axis ticks

    else:
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))  # 3 panels
        # Display the original image
        axes[0].imshow(img_show)
        axes[0].set_title("Original Image")
        axes[0].axis('off')  # Hide axis ticks

        # Display the prediction
        axes[1].imshow(pred_image, cmap=cmap)
        axes[1].set_title("Prediction")
        axes[1].axis('off')  # Hide axis ticks

        # Display the overlay of prediction on the original image
        axes[2].imshow(img_show)
        axes[2].imshow(pred_image, cmap=cmap, alpha=alpha)  # Overlay the prediction with transparency
        axes[2].set_title("Overlayed Prediction")
        axes[2].axis('off')  # Hide axis ticks

    # Save the figure
    plt.savefig(save_path)
    plt.close(fig)  # Close the figure to free memory

if __name__ == "__main__":

    dataset_name = "project2024"

    is_binary_classification = True

    if is_binary_classification: 
        task_name = "fcn"
        job_id = "132970"
        backbone = "resnet101"
        num_classes = 2
        classification_task = False
    else:
        job_id = "132727"
        backbone = "resnet101"
        task_name = "deeplab"
        num_classes = 30
        classification_task = True

    device = get_gpu()
    if device != None:
        aux = False
        pretrained = False
        
        if task_name == "fcn":
            model = create_fcn_model(backbone, 
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
            model = create_deeplab_model(backbone,
                                        aux=aux , 
                                        num_classes = num_classes, 
                                        pretrained = pretrained,
                                        classification_task = classification_task)
        else:
            raise ValueError("model setting error")

        model_file_path = f"{task_name}/{dataset_name}/module/{job_id}/model.pth"
        lr = 0.0001
        optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=1e-4)

        model, optimizer, epoch, additional_info = load_model_state(model_file_path, model, optimizer)
        filename = "test4"
        image_path = f'./test_images/{filename}.png'
        # mask_path = f'./test_images/{filename}_mask.png'
        image,mask = load_image(image_path)
        print(image.size())

        model.to(device)
        model.eval()
        with torch.no_grad():
            image = image.to(device)
            outputs = model(image)
            # 处理输出，例如获取预测结果
            _, preds = torch.max(outputs['out'], 1)
        

            visualize_image_and_prediction(image[0], preds[0], mask, './test_images/prediction_output.png')
    
        # perform evaluation on your validation/test dataset