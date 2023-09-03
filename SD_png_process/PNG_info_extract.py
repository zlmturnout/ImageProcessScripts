from PIL import Image
import os,sys,time,re

def get_prompt_png(png_file:str):
    if os.path.isfile(png_file) and png_file.endswith('png'):
        im = Image.open(filename)
        im.load()  # Needed only for .png EXIF data (see citation above)
        parameters=im.info.get('parameters',None)
        png_info_dict={"filename":png_file,"input_prompt":None,"negative_prompt":None}
        png_para_dict={}
        if parameters:
            input_paras=parameters.split('\n')
            png_info_dict["input_prompt"]=input_paras[0]
            png_info_dict["negative_prompt"]=re.match(r'Negative prompt[:](.*)',input_paras[1]).group(1)
            png_para_dict=text_to_dict(input_paras[-1])
        if not png_para_dict:
            png_info_dict.update(png_para_dict)
        png_info_dict["postprocessing"]=im.info.get('postprocessing',None)
        png_info_dict["extras"]=im.info.get('extras',None)
        png_info_dict["width"],png_info_dict["height"]=im.width,im.height
        
    return png_info_dict

def text_to_dict(text:str):
    """transform a text to dict data
    text_form=Steps: 20, Sampler: DPM++ 2S a, CFG scale: 7, Seed: 123839547, Face restoration: CodeFormer, Size: 512x768, Model hash: fc2511737a,
    Args:
        text (str): _description_
    """
    sep_text=text.split(', ')
    text_info_dict={}
    for item in sep_text:
        key=item.split(': ')[0]
        vars=item.split(': ')[-1]
        text_info_dict[key]=vars
    return text_info_dict

if __name__=="__main__":
    #filename = r'./img/00129-302883656.png'
    filename=r'./img/00037.png'
    # im = Image.open(filename)
    # im.load()  # Needed only for .png EXIF data (see citation above)
    png_info_dict=get_prompt_png(filename)
    print(png_info_dict)
    # parameters=im.info.get('parameters',None)

    # postprocessing=im.info.get('postprocessing',None)
    # extras=im.info.get('extras',None)
    # width,height=im.width,im.height
    # print(f'png {filename} with width,height={width}x{height}\n info:parameters={parameters}\npost processing={postprocessing}\nextras={extras}')