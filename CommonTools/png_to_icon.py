from PIL import Image
import os,sys,time

def png_to_icon(image_file:str,save_path:str,icon_sizes:tuple=[(128,128), (256, 256)]):
    if os.path.exists(image_file):
        folder,filename=os.path.split(image_file)
        filename_noext,ext=os.path.splitext(filename)
        save_path=save_path if save_path else folder
        #icon_sizes = [(16,16), (32, 32), (48, 48), (64,64)]
        icon_sizes = [(128,128), (256, 256)]
        if ext.lower() in ['.png','.jpg','.tif']:
            img = Image.open(image_file)
            new_logo_img=os.path.join(save_path,f'icon_{filename_noext}.ico')
            img.save(new_logo_img,sizes=icon_sizes)
            return new_logo_img

if __name__ == '__main__':
    img=os.path.abspath(r'./img/00072-1960626877.png')
    img_folder=r'K:\Coding\resource\imgs'
    for img in os.listdir(img_folder):
        img_abspath=os.path.join(img_folder,img)
        icon_path=os.path.join(os.path.dirname(img_abspath),'icon')
        if not os.path.exists(icon_path):
            os.mkdir(icon_path)
        ico_img=png_to_icon(img_abspath,icon_path)
        print(f'{img} -> {ico_img}')