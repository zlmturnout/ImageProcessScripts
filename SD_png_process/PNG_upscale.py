import time, datetime, os, sys
import cv2

def upscale_png(img_file:str,height:int=2160,width:int=3840,fit_on:int=0,scale_str:str="4K"):
    """resize the image to new height and width 

    Args:
        img_file (str):image file format (jpg,png,webm) 
        height (int): _description_
        width (int): _description_
        fit_on (int): fit on which dimension,0 is width,1 is height
    """
    #load img
    start_time=time.monotonic()
    img=cv2.imread(img_file)
    h,w=img.shape[0],img.shape[1]
    # scale=h/w
    # if scale>=height/width:
    #     new_img=cv2.resize(img,(width,int(h*width/w)),interpolation=cv2.INTER_CUBIC)
    # else:
    #     new_img=cv2.resize(img,(int(w*height/h),height))
    if fit_on==0:
        new_img=cv2.resize(img,(width,int(h*width/w)),interpolation=cv2.INTER_CUBIC)
    else:
        new_img=cv2.resize(img,(int(w*height/h),height))
    #save new img
    filename, extension=os.path.splitext(img_file)

    new_imgfile=filename+"_"+scale_str+extension
    cv2.imwrite(new_imgfile,new_img)
    print(f'image process finished in {time.monotonic()-start_time:.2f}s')
if __name__ == "__main__":
    img= r'./img/00187-3869624274.png'
    upscale_png(img,height=2160,width=3840,fit_on=1,scale_str='h2160')