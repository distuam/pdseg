import json
import os


def list_directories(folder_path:str):
    """
    返回指定路径下所有文件夹的名称。

    Parameters:
    - folder_path: 要搜索的文件夹的路径。

    Returns:
    - 一个包含所有文件夹名称的列表。
    """
    if not os.path.exists(folder_path):
        raise FileNotFoundError(f"The folder '{folder_path}' does not exist.")

    directories = []  # 用于存储文件夹名称的列表
    for entry in os.listdir(folder_path):
        entry_path = os.path.join(folder_path, entry)
        # 检查这个路径是否为文件夹
        if os.path.isdir(entry_path):
            directories.append(entry)

    return directories


def list_files_with_paths(folder_path: str , file_extensions: list, match: str):
    if not os.path.exists(folder_path):
        raise FileNotFoundError(f"The folder '{folder_path}' does not exist.")
    file_extensions = set(file_extensions)
    file_lists = [] 
    for file_name in os.listdir(folder_path):
        file_path = os.path.join(folder_path, file_name)
        if os.path.isfile(file_path):
            if match == 'start':
                if file_name.startswith(tuple(file_extensions)):
                    file_lists.append(file_name)
            if match == 'end':
                if file_name.endswith(tuple(file_extensions)):
                    file_lists.append(file_name)
    return file_lists

def find_files_with_extension(folder_path: str, file_extensions: list):
    file_list = []
    counter = 0
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            if file.endswith(tuple(file_extensions)):
                file_list.append(os.path.join(root, file))
                counter += 1
    print(counter)
    return file_list,counter


def write_file_paths_to_csv(csv_filename: str, file_list: list):
    print(csv_filename)
    with open(csv_filename, 'w') as csvfile:
        csvfile.write("file_path\n")
        for file_path in file_list:
            csvfile.write(file_path + '\n')


def path_exists(path: str) -> bool:
    """
    Check if the specified path exists.

    Args:
    - path (str): The path to be checked.

    Returns:
    - bool: True if the path exists, False otherwise.
    """
    return os.path.exists(path)

def read_json_file(json_file_path):
    with open(json_file_path, "r") as f:
        data = json.load(f)
    f.close()
    return data


def read_txt_file(file_path):
    """
    从txt文件中读取内容并返回列表。

    Args:
    - file_path (str): txt文件路径。

    Returns:
    - content_list (list): 文件内容列表，每一行作为一个元素。
    """
    content_list = []
    with open(file_path, "r") as file:
        content_list = [line.strip() for line in file.readlines()]
    return content_list

def make_paths(paths: list):
    for path in paths:
        os.makedirs(path, exist_ok=True)

def make_path(path: str):
    os.makedirs(path, exist_ok=True)