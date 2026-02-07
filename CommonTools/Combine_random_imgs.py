import os,sys,time,random
import numpy as np
from PIL import Image

class myTimer:
    def __init__(self, func):
        self.func = func

    def __call__(self, *args, **kwds):
        start_time = time.time()
        result=self.func(*args, **kwds)
        print(f'Function {self.func.__name__:*^20} cost: {time.time()-start_time:.4f}s\n')
        return result

# get all pic name
def get_IMG_list(dir: str, filelist: list):
    """
    find all image file in the dir
    :param dir: path to gallery
    :return:
    """
    for root, dirs, files in os.walk(dir):
        for file in files:
            if file.endswith(('.bmp', '.webp', '.png', '.jpg', '.jpeg')):
                filelist.append(os.path.join(root, file))
    return filelist 

def select_imgs(img_list,n):
    """select n random imgs from img_list
    image shoud have hight > width
    
    Args:
        img_list (_type_): _description_
        n (_type_): _description_
    """
    random_imgs=[]
    while len(random_imgs)<n:
        random_1=np.random.randint(0,len(img_list))
        img=Image.open(img_list[random_1])
        img_array=np.asarray(img)
        if img_array.shape[0]>img_array.shape[1]:
            random_imgs.append(img_list[random_1])
    return random_imgs

def resize_img_height(img_file,height:int=1800):
    img=Image.open(img_file)
    img_array=np.asarray(img)
    img_height,img_width=img_array.shape[:2]
    ratio=img_height/img_width
    new_width=int(height/ratio)
    img=img.resize((new_width,height))
    return img

def resize_img_width(img_file,width:int=1200):
    img=Image.open(img_file)
    img_array=np.asarray(img)
    img_height,img_width=img_array.shape[:2]
    ratio=img_width/img_height
    new_height=int(width/ratio)
    img=img.resize((width,new_height))
    return img

def combine_imgs_by_H(img_list,height:int=1800):
    """combine three images into one    """
    imgs=[]
    for img_file in img_list:
        img_norm=resize_img_height(img_file,height)
        imgs.append(img_norm)
    # combine the three images into one
    combined_img=np.concatenate(imgs,axis=1)
    return combined_img

def combine_imgs_by_W(img_list,width:int=1800):
    """combine three images into one    """
    imgs=[]
    for img_file in img_list:
        img_norm=resize_img_width(img_file,width)
        imgs.append(img_norm)
    # combine the three images into one
    combined_img=np.concatenate(imgs,axis=0)
    return combined_img

def random_5str():
    return ''.join(random.sample('abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789',5))

@myTimer
def combine_imgs_infolder(pic_folder,save_path:str,height:int=1800,all_num:int=30):
    img_list=get_IMG_list(pic_folder,[])
    i=0
    while i<all_num:
        random_imgs=select_imgs(img_list,3)
        combined_img=combine_imgs_by_H(random_imgs,height)
        timestamp=time.strftime("%Y%m%d%H%M", time.localtime())
        r_str=random_5str()
        Image.fromarray(combined_img).save(os.path.join(save_path,f"{str(i)}_{timestamp}_{r_str}.jpg"))
        i+=1
    return combined_img

if __name__=="__main__":
    pic_folder=r'E:\Open_pics\FlightAttendance'
    pic_folder=r'E:\Open_pics\方子萱'
    save_path=r'E:\Open_pics\CombinedIMGs'
    combine_imgs_infolder(pic_folder,save_path,all_num=30)