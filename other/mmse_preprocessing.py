# Author: Yunhan YE
# Date: 04/29/2024
# email: yunhan.ye@pootong.com or yunhan.ye@uq.edu.au
import os
import random
import shutil
import cv2
import numpy as np
import json
import path_handle
class convertImageToMask_In_MMSE():
    def __init__(self) -> None:
        self.classes_color = {} #这里保存{类：颜色}
        self.used_colors = set()  # 已使用的颜色集合
        self.next_gray_value = 0
    
    def set_label_path(self, label_path, output_path_name):
        self.is_binary_classification = False
        if path_handle.path_exists(label_path):
              
            self.output_path = os.path.dirname(os.path.abspath(label_path))# 获取label_path的上级文件夹路径
            # print(self.output_path)
            self.label_path = label_path

            self.output_path = self.output_path + f"/{output_path_name}"
            self.output_path = os.path.abspath(self.output_path)

            self.train_image_path = self.output_path + "/image//train"
            self.train_mask_path = self.output_path + "/mask//train"

            self.val_image_path = self.output_path + "/image//val"
            self.val_mask_path = self.output_path + "/mask//val"
            # print(self.train_label_path, self.val_label_path)

            path_handle.make_paths([self.train_image_path, self.val_image_path, self.train_mask_path, self.val_mask_path])
      
        else:
            raise FileNotFoundError("指定路径不存在！")
        
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

   
    def convert_label_to_mask(self, directory:str, directory_pull_path:str, output_mask_path: str, json_file_name: str):
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
                    mask_file_name = "bi_" + os.path.splitext(json_file_name)[0] + ".png" #二分类的图片名
                else:
                    mask_file_name = os.path.splitext(json_file_name)[0] + ".png"

                mask_file_path = os.path.join(output_mask_path, mask_file_name)
                cv2.imwrite(mask_file_path, mask)
                return mask_file_path

        
    def convert_labels_to_mask(self, num_classes, train_ratio: float = 0.85, seed:int = 42):
        gray_value = self.__get_gray_value("background")
        self.num_classes= num_classes
        self.train_ratio = train_ratio
        if self.check_config():
            self.directories = path_handle.list_directories(self.label_path)

            dict_length = len(self.directories)
            if num_classes < dict_length:
                # with open(image_txt_path, 'w') as image_file, open(label_txt_path, 'w') as label_file:
                if self.is_binary_classification:
                    prefix = "bi"
                else:
                    prefix = ""
                image_txt_path = os.path.join(self.output_path, f'{prefix}train_{num_classes}.txt')
                val_txt_path = os.path.join(self.output_path, f'{prefix}val_{num_classes}.txt')
                with open(image_txt_path, 'w') as image_file, open(val_txt_path, 'w') as val_file:
                    for directory in self.directories:
                        if self.next_gray_value < num_classes:
                            directory_pull_path = os.path.join(self.label_path, directory)
                            json_file_name_list = path_handle.list_files_with_paths(directory_pull_path, ["json"], "end")
                            jpg_file_name_list = path_handle.list_files_with_paths(directory_pull_path, ["jpg","png"], "end")

                            print(directory, len(json_file_name_list) )

                            if len(json_file_name_list) <= 0:
                                print("jump:", directory_pull_path)
                            else:
                                length_json_list = len(json_file_name_list)#这里只能使用json分割，因为有些image没有制作mask
                                print(len(json_file_name_list),len(jpg_file_name_list))
                                # 打乱列表前先创建索引列表
                                random.seed(seed)
                                random.shuffle(json_file_name_list)
                                
                                # 计算分割点
                                split_index = int(train_ratio * length_json_list)
                                splited_json_list = [(0, json_file_name_list[:split_index]), (1, json_file_name_list[split_index:])]
                    

                                for index, json_file_name_list in splited_json_list:
                                    if index == 0:
                                        output_mask_path = self.train_mask_path
                                    else:
                                        output_mask_path = self.val_mask_path
                                    for json_file_name in json_file_name_list:
                                        json_file_base_name = os.path.splitext(json_file_name)[0]
                                        corresponding_jpg_file = json_file_base_name + ".jpg"

                                        if corresponding_jpg_file in jpg_file_name_list:
                                            mask_file_path = self.convert_label_to_mask(directory, directory_pull_path, output_mask_path, json_file_name)
                                            if mask_file_path != None:
                                                if index == 0: #训练
                                
                                                    if self.is_binary_classification:
      
                                                        new_filename = f"bi_{corresponding_jpg_file}"
                                                        work_path = os.path.join(self.train_image_path, new_filename)
                                                        json_file_base_name  = f"bi_{json_file_base_name}"
                                                    else:
                                                        work_path = self.train_image_path

                                                    shutil.copy(directory_pull_path + f"/{corresponding_jpg_file}", work_path)

                                                    image_file.write(f'{json_file_base_name}\n')
                                                else:
                                                    if self.is_binary_classification:
                                                        new_filename = f"bi_{corresponding_jpg_file}"
                                                        work_path = os.path.join(self.val_image_path, new_filename)
                                                        json_file_base_name  = f"bi_{json_file_base_name}"
                                                    else:
                                                        work_path = self.val_image_path

                                                    shutil.copy(directory_pull_path + f"/{corresponding_jpg_file}", work_path)

                                                    val_file.write(f'{json_file_base_name}\n')
                                            jpg_file_name_list.remove(corresponding_jpg_file)
                                if self.is_binary_classification: #因为二分类不在程序中创建颜色则，在这里限制颜色
                                    self.next_gray_value += 1
                        else:
                            break
                image_file.close()
                val_file.close()
            else:
                raise ValueError("超过类的最大数量")
            self.__output_config_file()
    
    def check_config(self):
        # try:
        #     config = path_handle.read_json_file(self.output_path + f"//config_{self.num_classes}.json")
        #     if config["number_of_classes"] == self.num_classes:
        #         print(f"当前类的文件夹已经生成")
        #         return False
        #     else:
        #         shutil.rmtree(self.output_path)
        #         return True
        # except Exception as e:
        #     print(f"An error occurred while reading config file: {e}")
        # shutil.rmtree(self.output_path)
        return True


    def __output_config_file(self):
        if self.is_binary_classification:
            file_name = f"biconfig_{self.num_classes}.json"
            self.classes_color["plant_diseases"] = 1
        else:
            file_name = f"config_{self.num_classes}.json"

        config_data = {
            "train_ratio": self.train_ratio, #训练集占比
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

    convertImageToMask_class = convertImageToMask_In_MMSE()
    convertImageToMask_class.set_label_path(jsonfileFolder, "project2024_copy")
    convertImageToMask_class.set_is_binary_classification(True)
    convertImageToMask_class.convert_labels_to_mask(num_classes=12)


