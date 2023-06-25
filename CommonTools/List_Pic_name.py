# -*- coding: utf-8 -*-
import os, sys, time, datetime,re

MainPath = "I:\\Coding\\Pycoding\\Cynthia\\Pic_process\\"
pic_folder = os.path.join(MainPath, "sample")

def get_datetime():
    """ get current date time, as accurate as milliseconds

        Args: None

        Returns:
            str type
            eg: "2018-10-01 00:32:39.993176"

    """
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
    #print(timestamp)
    # return str(datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f'))
    return timestamp

# get all pic name
def get_IMG_list(dir: str, filelist: list):
    """
    find all image file in the dir
    :param dir: path to gallery
    :return:
    """
    for pic in os.listdir(dir):
        absPath = os.path.join(dir, pic)
        if os.path.isfile(absPath):
            if pic.endswith(('.bmp', '.webp', '.png', '.jpg', '.jpeg')):
                filelist.append(pic)
    print(filelist)
    return filelist


def generate_img_text(folder_path: str, relative_path: str='/'):
    """
    generate a txt file list all pics with relative_path
    :param relative_path: relative path like "/img/"
    :param folder_path: path to the pic folder
    :return:
    """
    if os.path.isdir(folder_path):
        pic_list=get_IMG_list(folder_path,[])
        txt_file = os.path.join(folder_path, 'img_list.txt')
        print(txt_file)
        timestamp = get_datetime()
        with open(txt_file,'w+',encoding='utf-8') as fp:
            fp.writelines(f'absolute img path: {folder_path}\ntimestamp: {timestamp}\n')
            for index,each_pic in enumerate(pic_list):

                # re_path=os.path.join(relative_path,each_pic)
                re_path=f'"url({relative_path}{each_pic})",'
                print(f'{index}. {re_path}')
                fp.write(re_path+'\n')

if __name__ == "__main__":
    pic_folder="D:/MyBlog/limin_blog/source/Gallery/BG_img"
    generate_img_text(pic_folder,'/Gallery/BG_img/')