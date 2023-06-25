# coding=utf-8
import time, datetime, os, sys
import cv2


def create_path(file_path):
    if not os.path.exists(file_path):
        os.makedirs(file_path)
    os.chdir(file_path)
    return file_path

def resize4K_img(img_file:str,height:int=2160,width:int=3840):
    """resize the image to new height and width 

    Args:
        img_file (str):image file format (jpg,png,webm) 
        height (int): _description_
        width (int): _description_
    """
    #load img
    start_time=time.monotonic()
    img=cv2.imread(img_file)
    h,w=img.shape[0],img.shape[1]
    scale=h/w
    if scale>=height/width:
        new_img=cv2.resize(img,(width,int(h*width/w)),interpolation=cv2.INTER_CUBIC)
    else:
        new_img=cv2.resize(img,(int(w*height/h),height))
    #save new img
    filename, extension=os.path.splitext(img_file)

    new_imgfile=filename+"_4K"+extension
    cv2.imwrite(new_imgfile,new_img)
    print(f'image process finished in {time.monotonic()-start_time:.2f}s')

if __name__ == "__main__":
    img="C://Users//Limin  Zhou//Pictures//AI_wallpaper//flowers.png"
    resize4K_img(img)

