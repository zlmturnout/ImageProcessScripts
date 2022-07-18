# coding=utf-8
import time, datetime, os, sys
import shutil
from tkinter import E

def create_path(file_path):
    if not os.path.exists(file_path):
        os.makedirs(file_path)
    os.chdir(file_path)
    return file_path


def extract_pic_folder(src_path, dst_path,filetype=None):
    """
    Extract and rename file of given filetype in the src_path to dst_path(one folder)
    :param src_path:
    :param dst_path:
    :param filetype: list=['.jpg','.png','.tiff','.webp','.png']
    :return:
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
    upper_path, folder_name = os.path.split(dst_path)
    if not os.listdir(src_path):
        return
    else:
        for item in os.listdir(src_path):
            folder_path = os.path.join(src_path, item)
            if os.path.isdir(folder_path):
                # 分解得到文件夹名字
                """
                upper_path,folder_name=os.path.split(abs_folder_path)
                """
                upper_path, folder_name = os.path.split(folder_path)
                # index all matched pics
                index_pic=0
                for item in os.listdir(folder_path):
                    # 分解得到文件名和后缀
                    """
                    filename,extension=os.path.splitext(abs_filepath)
                    """
                    filename, extension = os.path.splitext(item)
                    if extension in filetype:
                        # origin image abs_path
                        origin_pic_path = os.path.join(folder_path,item)
                        All_pics_N += 1
                        # create new image  folder and get abs_path
                        save_folder_path=create_path(os.path.join(dst_path,folder_name))
                        new_pic_path=os.path.join(save_folder_path,str(All_pics_N)+extension)
                        print(f'origin_pic: {origin_pic_path},new_pic: {new_pic_path}')
                        # copy image file to dst_path/folder_name
                        #shutil.copy(origin_pic_path, new_pic_path)
                        print(f'move origin_pic: {origin_pic_path} to new path\n new_pic: {new_pic_path}')
                        index_pic += 1

    ## full pic
    print(f'number of pics: {All_pics_N}')

                
if __name__ == "__main__":
    download_path = r'F:\BeautifulPictures\NET'
    save_path = r'F:\Perilous\IMGs'
    extract_pic_folder(download_path,save_path)
    ## full pic
    