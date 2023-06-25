# -*- coding: utf-8 -*-
import os, sys, time, datetime,re

MainPath = "I:\\Coding\\Pycoding\\Cynthia\\Pic_process\\"
folder = os.path.join(MainPath, "sample")


def get_datetime():
    """ get current date time, as accurate as milliseconds

        Args: None

        Returns:
            str type
            eg: "2018-10-01 00:32:39.993176"

    """
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
    print(timestamp)
    # return str(datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f'))
    return timestamp


def rename_pics(folder):
    """
    find the image file and rename it
    :param folder: path to gallery
    :return:
    """
    i = 0
    for pic in os.listdir(folder):
        absPath = os.path.join(folder, pic)
        if os.path.isfile(absPath):
            if pic.endswith(('.bmp', '.gif', '.png', '.jpg', '.jpeg', '.tif')):
                name, ext = os.path.splitext(pic)
                newPath = os.path.join(folder, str(i) + ext)
                print(newPath)
                # rename the pic file
                if absPath != newPath:
                    os.rename(absPath, newPath)
                i += 1
        else:
            print("not file")


# get all pic name
def getFilelist(dir: str, filelist: list):
    """
    find all image file in the dir
    :param dir: path to gallery
    :return:
    """
    for pic in os.listdir(dir):
        absPath = os.path.join(dir, pic)
        if os.path.isfile(absPath):
            if pic.endswith('.jpg'):
                filelist.append(pic)
    print(filelist)
    return filelist


#  write markdown file md
def md_generator(galleryPath: str, category: str = "Gallery", tags: str = "beauty"):
    """
    generate a md file according to the images in the gallery
    :param tags: set tags for the gallery
    :param category: set category
    :param galleryPath: path to gallery
    :return:
    """
    picNames = getFilelist(galleryPath, [])
    galleryName = os.path.basename(galleryPath)
    gallery=re.sub('[^\w\s]', '', galleryName)
    md_file = os.path.join(os.path.dirname(galleryPath), galleryName + '.md')
    print(md_file)
    timestamp = get_datetime()
    with open(md_file, 'w+',encoding='utf-8') as fp:
        font_note = f'---\rtitle: {gallery}  \rcover: /{picNames[0]}\rdate: {timestamp}\rcategories:\n- {category}\r' \
                    f'tags:\n- {tags}\r  ---\n\n'
        fp.write(font_note)
        fp.write(f'# Gallery\n+ **{galleryName}**\n')
        # pic numbers
        fp.write(f'**All pics: {len(picNames)-1}+1p**\n')
        for pic in picNames[1:]:
            fp.write(f'![]({pic})\n')



if __name__ == "__main__":
    # rename_pics(folder)
    # getFilelist(folder, [])
    # md_generator(folder)
    path="F:\\"
    for dir in os.listdir(path):
        absdir=os.path.join(path,dir)
        md_generator(absdir)
