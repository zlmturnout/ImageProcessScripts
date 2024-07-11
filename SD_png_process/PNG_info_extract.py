from PIL import Image
import os,sys,time,re,json

def get_prompt_png(png_file:str):
    if os.path.isfile(png_file) and png_file.endswith('png'):
        im = Image.open(png_file)
        im.load()  # Needed only for .png EXIF data (see citation above)
        parameters=im.info.get('parameters',None)
        png_info_dict={"filename":png_file,"input_prompt":None,"negative_prompt":None}
        png_para_dict={}
        if parameters:
            input_paras=parameters.split('\n')
            png_info_dict["input_prompt"]=input_paras[0]
            png_info_dict["negative_prompt"]=re.match(r'Negative prompt[:](.*)',input_paras[1]).group(1)
            png_para_dict=text_to_dict(input_paras[-1])
        png_info_dict.update(png_para_dict)
        png_info_dict["postprocessing"]=im.info.get('postprocessing',None)
        png_info_dict["extras"]=im.info.get('extras',None)
        png_info_dict["width"],png_info_dict["height"]=im.width,im.height
        # save to txt file
        save_folder=os.path.dirname(png_file)
        txt_name=os.path.splitext(os.path.basename(png_file))[0]+'.txt'
        dict_to_text(png_info_dict,save_folder,txt_name)
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

def dict_to_text(data_dict: dict, file_path: str, file_name: str):
    """
    Transform dict form data to text form and save to file
    :param data_dict:
    :param file_path:
    :param file_name:
    :return:
    """
    txt_file=os.path.join(file_path,file_name)
    with open(txt_file, 'w') as f:
        for k, v in data_dict.items():
            f.write(k + ': ' + str(v) + '\n')
    return txt_file

def comfy_ui_png_info(png_file:str):
    if os.path.isfile(png_file) and png_file.endswith('png'):
        im = Image.open(png_file)
        im.load()  # Needed only for .png EXIF data (see citation above)
        prompt_json=json.loads(im.info.get('prompt',None))
        workflow_json=json.loads(im.info.get('workflow',None))
        # for comfy ui 2x 
        if '8' in prompt_json.keys():
            pos_prompt=prompt_json['8']['inputs']['positive']
            neg_prompt=prompt_json['8']['inputs']['negative']
            model=prompt_json['8']['inputs']['base_ckpt_name']
        elif '4' in prompt_json.keys():
            pos_prompt=prompt_json['4']['inputs']['positive']
            neg_prompt=prompt_json['4']['inputs']['negative']
            model=prompt_json['4']['inputs']['base_ckpt_name']
        return model,pos_prompt,neg_prompt
    
if __name__=="__main__":
    #filename = r'./img/00129-302883656.png'
    filename=r'F:\Coding\PythonProjects\ImageProcessScripts\img\00002-3558247218.png'
    filename2=r'./img/SDXL__00001_.png'
    
    im = Image.open(filename2)
    im.load()  # Needed only for .png EXIF data (see citation above)
    prompt_json=json.loads(im.info.get('prompt',None))
    workflow_json=json.loads(im.info.get('workflow',None))
    # for comfy ui
    # pos_prompt=prompt_json['8']['inputs']['positive']
    # neg_prompt=prompt_json['8']['inputs']['negative']
    # model=prompt_json['8']['inputs']['base_ckpt_name']
    model,pos_prompt,neg_prompt=comfy_ui_png_info(filename2)
    print(f'model:\n{model}\npos_prompt:\n{pos_prompt}\nneg_prompt:\n{neg_prompt}')
    # for SD-web-ui
    parameters=im.info.get('parameters',None)
    png_info_dict=get_prompt_png(filename)
    print(png_info_dict["input_prompt"])
    # postprocessing=im.info.get('postprocessing',None)
    # extras=im.info.get('extras',None)
    # width,height=im.width,im.height
    # print(f'png {filename} with width,height={width}x{height}\n info:parameters={parameters}\npost processing={postprocessing}\nextras={extras}')