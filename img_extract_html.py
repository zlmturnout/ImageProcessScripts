# coding=utf-8
import time, datetime, os, sys
import shutil


def create_path(file_path):
    if not os.path.exists(file_path):
        os.makedirs(file_path)
    os.chdir(file_path)
    return file_path


def copy_pic_folder(src_path, dst_path, filetype=None):
    """
    copy and rename file of given filetype in the src_path to dst_path
    :param src_path:
    :param dst_path:
    :param filetype: list=['.jpg','.png','.tiff','.webp','.png']
    :return:
    """
    if filetype is None:
        filetype = ['.jpg', '.png', '.tiff', '.webp', '.png']
    if not os.path.isdir(src_path):
        print(f'{src_path} is not a valid path!')
        return
    if not os.path.isdir(dst_path):
        dst_path = os.getcwd()
    save_path = create_path(dst_path)
    list_dir = os.listdir(src_path)
    num_pics = 0
    for i in range(0, len(list_dir)):
        folder_path = os.path.join(src_path, list_dir[i])
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
                    num_pics += 1
                    # create new image  folder and get abs_path
                    save_folder_path=create_path(os.path.join(dst_path,folder_name))
                    new_pic_path=os.path.join(save_folder_path,str(index_pic)+extension)
                    print(f'origin_pic: {origin_pic_path},new_pic: {new_pic_path}')
                    # copy image file to dst_path/folder_name
                    shutil.copy(origin_pic_path, new_pic_path)
                    print(f'move origin_pic: {origin_pic_path} to new path\n new_pic: {new_pic_path}')
                    index_pic += 1
    ## full pic
    print(f'number of pics: {num_pics}')


if __name__ == "__main__":
    download_path = r'C:\Users\Limin  Zhou\Downloads\T0'
    save_path = r'F:\Perilous\IMGs'
    #pic_path = os.path.join(download_path, '23.jpg')
    # print(pic_path)
    # new_pic_path=os.path.join(save_path,'23.jpg')
    # print(new_pic_path)
    # #os.rename(pic_path,new_pic_path)
    # shutil.copy(pic_path,new_pic_path)
    list_pics = os.listdir(download_path)
    print(len(list_pics))
    print(os.path.isdir(download_path))
    copy_pic_folder(download_path, save_path)
