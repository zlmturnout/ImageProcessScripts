import os,sys
from PIL import Image

def conert_pngToicon(folder_path:str):
    """convert any image[png,jpg,tif..] to windows iconfile

    Args:
        folder_path (str): _description_
    """
    imgtypes = ['.jpg', '.png', '.tiff',  '.png']
    icon_size=(256,256)
    if os.path.isdir(folder_path):
        icon_path=os.path.join(folder_path,'icon')
        if not os.path.exists(icon_path):
            os.mkdir(icon_path)
        for file in os.listdir(folder_path):
            filename,extension=os.path.splitext(file)
            if extension in imgtypes:
                img=Image.open(os.path.join(folder_path,file)).resize(icon_size)
                try:
                    icon_filepath=os.path.join(icon_path,f'{filename}.ico')
                except IOError:
                    print(f'can not convert:{file}')
                else:
                    img.save(icon_filepath,format="ICO")
                    print(f'convert {file} to iconfile finished')

if __name__ == "__main__":
    pic_folder=r"H:\\imgs"
    conert_pngToicon(pic_folder)       
            