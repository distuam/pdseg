# Author: Yunhan YE
# Date: 04/07/2024
# email: yunhan.ye@pootong.com or yunhan.ye@uq.edu.au
import os
import cv2
import numpy as np
import json
import path_handle

class convertImageToMask():
    def __init__(self) -> None:
        self.classes_color = {} #这里保存{类：颜色}
        self.used_colors = set()  # 已使用的颜色集合
        self.next_gray_value = 0
        self.is_binary_classification = False
    
    def set_label_path(self, label_path):
        if path_handle.path_exists(label_path):
            label_path = os.path.abspath(label_path)
            self.label_path = label_path
            self.output_path = label_path
        else:
            raise FileNotFoundError("指定路径不存在！")
    
    def get_label_path(self):
        return self.label_path
    
    def set_output_path(self, output_path):
        if path_handle.path_exists(output_path):
            self.output_path = os.path.abspath(output_path)
        
    
    def get_output_path(self):
        return self.output_path
    
    def set_is_binary_classification(self, is_binary_classification):
        self.is_binary_classification = is_binary_classification


    def __get_color_for_class(self, class_name):
        if class_name not in self.classes_color:
            # 如果类名不在字典中，则生成新颜色并添加到字典中
            new_color = self.__generate_new_color()
            self.classes_color[class_name] = new_color
            self.used_colors.add(new_color)
            return new_color
        return self.classes_color[class_name]
    
    def __generate_new_color(self):
        # 生成一个新颜色，确保它与已使用的颜色不重复
        while True:
            color = tuple(np.random.randint(0, 256, size=3).tolist())
            if color not in self.used_colors:
                return color
    
    def __get_gray_value(self, class_name):
        if class_name not in self.classes_color:
            # 分配一个新的灰度值给这个类别
            print(class_name, self.next_gray_value)
            self.classes_color[class_name] = self.next_gray_value
            self.next_gray_value += 1
           
            # 确保灰度值不超过255
            if self.next_gray_value > 255:
                raise ValueError("超出了可分配的灰度值范围")
        return self.classes_color[class_name]

   
    def convert_label_to_mask(self,directory:str, directory_pull_path: str, json_file_name: str):
        #directory 可以作为类名判断标记是否属于类名
        json_file_path = os.path.join(directory_pull_path, json_file_name)
        # print(json_file_path)
        if os.path.getsize(json_file_path) == 0:
            os.remove(json_file_path)
            return None
        else:
            with open(json_file_path, "r", encoding='utf-8') as jsonf:
                jsonData = json.load(jsonf)
                img_h = jsonData["imageHeight"]
                img_w = jsonData["imageWidth"]
                if img_h > 2048 and img_w > 2048:
                    print("Big image: ",json_file_path)
                    return None
                else:
                    # mask_line = np.zeros((img_h, img_w, 3), np.uint8)  # 假定一个3通道图像用于彩色掩码
                    mask = np.zeros((img_h, img_w), dtype=np.uint8)  # 使用单通道图像
                    
                    # mask = np.full((img_h, img_w), 255, dtype=np.uint8)
                    for obj in jsonData["shapes"]:
                        label = obj["label"]
                        if label != directory:
                            print("Warning: Label mismatch - Expected:", directory, "Actual:", label)
                            continue

                        # color = self.__get_color_for_class(label)  # 获取或生成颜色
                        # polygonPoints = np.array(obj["points"], np.int32)
                        # cv2.drawContours(mask_line, [polygonPoints], -1, color, 3)
                        if self.is_binary_classification: #二分类的图片颜色
                            gray_value = 1
                        else:
                            gray_value = self.__get_gray_value(label)
                            
                        polygonPoints = np.array(obj["points"], np.int32)
                        cv2.fillPoly(mask, [polygonPoints], color=gray_value)

                    if self.is_binary_classification:
                        mask_file_name = "bi_" + os.path.splitext(json_file_name)[0] + "_mask.png" #二分类的图片名
                    else:
                        mask_file_name = os.path.splitext(json_file_name)[0] + "_mask.png"
                    mask_file_path = os.path.join(directory_pull_path, mask_file_name)
                    cv2.imwrite(mask_file_path, mask)
                    return mask_file_path

        
    def convert_labels_to_mask(self, num_classes:int):
        gray_value = self.__get_gray_value("background")

        #这里用来增加index
        self.directories = path_handle.list_directories(self.label_path)

        dict_length = len(self.directories)

        if num_classes < dict_length:
            if self.is_binary_classification:
                prefix = "bi"
            else:
                prefix = ""
            image_txt_path = os.path.join(self.label_path, f'{prefix}image_{num_classes}.txt')
            label_txt_path = os.path.join(self.label_path, f'{prefix}label_{num_classes}.txt')
            cls_task_label_txt_path = os.path.join(self.label_path, f'{prefix}cls_task_label_{num_classes}.txt')
            with open(image_txt_path, 'w') as image_file, open(label_txt_path, 'w') as label_file, open(cls_task_label_txt_path, 'w') as cls_task_label_file:
                for directory in self.directories:
                    # print(self.next_gray_value, num_classes, self.next_gray_value < num_classes)
                    if self.next_gray_value < num_classes:
                        directory_pull_path = os.path.join(self.label_path, directory)
                        json_file_name_list = path_handle.list_files_with_paths(directory_pull_path, ["json"], "end")
                        jpg_file_name_list = path_handle.list_files_with_paths(directory_pull_path, ["jpg","png"], "end")
                        # print(directory_pull_path, len(json_file_name_list) )
                        if len(json_file_name_list) <= 0:
                            print("jump:", directory_pull_path)
                        else:
                            for json_file_name in json_file_name_list:
                                json_file_base_name = os.path.splitext(json_file_name)[0]
                                corresponding_jpg_file = json_file_base_name + ".jpg"

                                if corresponding_jpg_file in jpg_file_name_list:
                                    mask_file_path = self.convert_label_to_mask(directory, directory_pull_path, json_file_name)
                                    if mask_file_path != None:
                                        image_file.write(os.path.join(directory_pull_path, corresponding_jpg_file) + '\n')
                                        label_file.write(os.path.join(directory_pull_path, mask_file_path) + '\n')
                                        if self.is_binary_classification:#二分类 制作标签没有意义，因为样本中没有正常的照片
                                            cls_task_label_file.write("1\n")
                                        else:
                                            cls_task_label_file.write(f"{self.classes_color[directory]}\n") #分类任务记录当前的类,记录当前的颜色，如果是类名修改为directory
                                    jpg_file_name_list.remove(corresponding_jpg_file)

                            if self.is_binary_classification: #因为二分类不在程序中创建颜色则，在这里限制颜色
                                self.next_gray_value += 1
                    
                    else:
                        break
            image_file.close()
            label_file.close()
            self.__output_config_file(num_classes)
        else:
            raise ValueError("超过类的最大数量")

    def __output_config_file(self,num_classes):
        if self.is_binary_classification:
            file_name = f"biconfig_{num_classes}.json"
            self.classes_color["plant_diseases"] = 1
        else:
            file_name = f"config_{num_classes}.json"


        config_data = {
            "number_of_classes": len(self.classes_color),  # 类的数量
            "class_names": list(self.classes_color.keys()),  # 类的名称
            "class_colors": self.classes_color  # 类对应的颜色或灰度值
        }
 
       
        config_path = os.path.join(self.output_path, file_name)
        with open(config_path, 'w', encoding='utf-8') as config_file:
            json.dump(config_data, config_file, ensure_ascii=False, indent=4)

        print(f"配置文件已生成在：{config_path}")

    def display_mask(self, mask):
        cv2.imshow("Mask", mask)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

if __name__ == "__main__":
    jsonfileFolder = r"../data/project2024/"

    convertImageToMask_class = convertImageToMask()
    convertImageToMask_class.set_label_path(jsonfileFolder)
    convertImageToMask_class.set_is_binary_classification(True)
    convertImageToMask_class.convert_labels_to_mask(num_classes=30)



