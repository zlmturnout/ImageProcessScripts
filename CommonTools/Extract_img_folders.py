# coding=utf-8
import time, datetime, os, sys
import shutil
from tkinter import E

def create_path(file_path):
    if not os.path.exists(file_path):
        os.makedirs(file_path)
    os.chdir(file_path)
    return file_path


def extract_pic_folder(src_path, dst_path,pic_list,filetype=None):
    """
    Extract and rename file of given filetype in the src_path to dst_path(one folder)
    :param src_path:
    :param dst_path:
    :param filetype: list=['.jpg','.png','.tiff','.webp','.png']
    :return:
        [list of pics ]: list of all pics in the directory(not subfolder)
    """
    All_pics_N=0
    if filetype is None:
        filetype = ['.jpg', '.png', '.tiff', '.webp', '.png']
    if not os.path.isdir(src_path):
        print(f'{src_path} is not a valid path!')
        return
    if not os.path.isdir(dst_path):
        dst_path = os.getcwd()
    save_path = create_path(dst_path)
    upper_path, folder_name = os.path.split(src_path)
    if not os.listdir(src_path):
        return []
    else:
        for item in os.listdir(src_path):
            All_pics_N+=1
            folder_path = os.path.join(src_path, item)
            if os.path.isfile(folder_path):
                filename, extension = os.path.splitext(item)
                if extension in filetype:
                    # origin image abs_path
                    origin_pic_path = folder_path
                    All_pics_N += 1
                    # create new image  folder and get abs_path
                    save_folder_path=create_path(os.path.join(dst_path,folder_name))
                    new_pic_path=os.path.join(save_folder_path,str(All_pics_N)+extension)
                    print(f'origin_pic: {origin_pic_path},new_pic: {new_pic_path}')
                    #copy image file to dst_path/folder_name
                    shutil.copy(origin_pic_path, new_pic_path)
                    print(f'move origin_pic: {origin_pic_path} to new path\n new_pic: {new_pic_path}')
                    pic_list.append(item)
            elif os.path.isdir(folder_path):
                extract_pic_folder(folder_path,dst_path,pic_list)
    return pic_list

def extract_pics(src_path,dst_path,filetype=None):
    """extract every pic in the folder to another folder with index 

    Args:
        src_path (_type_): _description_
        dst_path (_type_): _description_
        filetype (_type_, optional): _description_. Defaults to None.
    """
    All_pics_N=0
    if filetype is None:
        filetype = ['.jpg', '.png', '.tiff', '.webp', '.png']
    if not os.path.isdir(src_path):
        print(f'{src_path} is not a valid path!')
        return
    if not os.path.isdir(dst_path):
        dst_path = os.getcwd()
    for root, dirs, files in os.walk(src_path, topdown=True):
        for item in files:
            filename, extension = os.path.splitext(item)
            if extension in filetype:
                All_pics_N += 1
                # create new image  folder and get abs_path
                origin_pic_path=os.path.join(root, item)
                new_pic_path=os.path.join(dst_path,str(All_pics_N)+extension)
                shutil.copy(origin_pic_path, new_pic_path)
    print(f'all pics num: {All_pics_N}')
            

                
if __name__ == "__main__":
    #download_path = r'F:\BeautifulPictures'
    download_path=r'N:\XImgs\tuli\niannian'
    save_path = r'N:\XImgs\niannian'
    save_path = create_path(save_path)
    #pic_list=extract_pic_folder(download_path,save_path,[])
    #print(f'get all pics:{pic_list} with number {len(pic_list)}')
    extract_pics(download_path,save_path)