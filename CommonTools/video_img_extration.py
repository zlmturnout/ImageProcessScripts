import subprocess
import os

def ffmpeg_extract_frames(video_path, output_folder):
    # 确保输出文件夹存在
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # 构建FFmpeg命令
    command = [
        'ffmpeg',
        '-i', video_path,  # 输入视频文件
        '-vf', 'fps=1',  # 每秒提取一帧
        os.path.join(output_folder, 'frame_%04d.png')  # 输出文件名格式
    ]

    # 执行FFmpeg命令
    subprocess.run(command, check=True)
import subprocess
import os

def ffmpeg_tif_to_png(tif_folder, output_folder):
    """
    将TIFF图片文件夹中的图片转换为PNG格式。

    :param tif_folder: TIFF图片文件夹路径
    :param output_folder: 输出文件夹路径
    """
    # 获取TIFF图片文件夹中的所有图片文件
    tif_files = [os.path.join(tif_folder, f) for f in os.listdir(tif_folder) if f.endswith('.tif')]
    
    # 检查是否有TIFF图片文件
    if not tif_files:
        print("TIFF图片文件夹中没有TIFF图片文件。")
        return
    
    # 确保输出文件夹存在
    os.makedirs(output_folder, exist_ok=True)
    
    # 遍历TIFF图片文件
    for tif_file in tif_files:
        # 构建输出文件路径
        output_file = os.path.join(output_folder, os.path.splitext(os.path.basename(tif_file))[0] + '.png')
        
        # 构建FFmpeg命令
        command = [
            'ffmpeg',
            '-i', tif_file,
            output_file
        ]
        
        # 执行FFmpeg命令
        subprocess.run(command, check=True)
        print(f"已转换：{tif_file} -> {output_file}")


def ffmpeg_images_to_video(image_folder, output_video, fps=24):
    """
    将图片文件夹中的图片拼接成视频。

    :param image_folder: 图片文件夹路径
    :param output_video: 输出视频文件路径
    :param fps: 帧率
    """
    # 获取图片文件夹中的所有图片文件
    image_files = [os.path.join(image_folder, f) for f in os.listdir(image_folder) if f.endswith(('.png', '.jpg', '.jpeg'))]
    
    # 检查是否有图片文件
    if not image_files:
        print("图片文件夹中没有图片文件。")
        return
    
    # 对图片文件进行排序
    image_files.sort()
    image_file_txt=os.path.join(image_folder, 'image_files.txt')
    with open(image_file_txt, 'w+') as f:
        for idx,image_file in enumerate(image_files):
            new_img_file=os.path.join(image_folder, f'{idx}.png')
            os.rename(image_file, new_img_file)
            f.write(f'{idx}.png\n')
    
    # 构建FFmpeg命令
    command = [
        'ffmpeg',
        '-framerate', str(fps),
        '-f', 'image2',
        '-i', f'{image_folder}\%d.png',
        '-c:v', 'libx264',
        '-profile:v', 'high',
        '-r', str(fps),
        '-pix_fmt', 'yuv420p',
        '-vf', f'fps={fps}',
        output_video
    ]
    
    # 执行FFmpeg命令
    subprocess.run(command, check=True)
    print(f"视频已生成：{output_video}")


def ffmpeg_images_to_gif(image_folder, output_gif, fps=10):
    """
    将图片文件夹中的图片拼接成GIF动画。

    :param image_folder: 图片文件夹路径
    :param output_gif: 输出GIF文件路径
    :param fps: 帧率
    """
    # 获取图片文件夹中的所有图片文件
    image_files = [os.path.join(image_folder, f) for f in os.listdir(image_folder) if f.endswith(('.png', '.jpg', '.jpeg'))]
    
    # 检查是否有图片文件
    if not image_files:
        print("图片文件夹中没有图片文件。")
        return
    
    # 对图片文件进行排序
    image_files.sort()
    
    # 构建FFmpeg命令
    command = [
        'ffmpeg',
        '-framerate', str(fps),
        '-i', f'{image_folder}\%d.png',
        '-vf', f'fps={fps}',
        '-loop', '0',
        output_gif
    ]
    
    # 执行FFmpeg命令
    subprocess.run(command, check=True)
    print(f"GIF动画已生成:{output_gif}")


def ffmpeg_cut_mp4(input_file, output_file, start_time, duration):
    """
    使用FFmpeg从源媒体中剪切视频。

    参数:
    input_file (str): 输入视频文件的路径。
    output_file (str): 输出视频文件的路径。
    start_time (str): 剪切开始时间，格式为 "HH:MM:SS"。
    duration (str): 剪切持续时间，格式为 "HH:MM:SS"。
    """
    command = [
        'ffmpeg',
        '-i', input_file,
        '-ss', start_time,
        '-t', duration,
        '-c', 'copy',
        output_file
    ]
    subprocess.run(command, check=True)


# 示例用法
if __name__ == '__main__':
    video_path = os.path.abspath(r'J:\Films\201210_0701_1080P_4000K_378043122.mp4')
    output_folder = os.path.join(os.path.dirname(video_path), 'frames')
    #create output_folder
    #extract_frames(video_path, output_folder)
    print('视频帧提取完成!')
    
    tif_folder= os.path.abspath(r'X:\DataAnalysis2024-06\bubble_1d8H2')
    png_folder = os.path.join(os.path.dirname(tif_folder), os.path.basename(tif_folder)+'_png')
    ffmpeg_tif_to_png( tif_folder, png_folder)
    # png to video
    #image_folder = os.path.abspath(r'I:\Coding\GitReposity\ImageProcessScripts\img\bubble_1d8H2_png')
    output_video=os.path.join(png_folder , 'output_video.mp4')
    ffmpeg_images_to_video(png_folder, output_video)
    # png to gif
    output_gif=os.path.join(png_folder , 'output_gif.gif')
    ffmpeg_images_to_gif(png_folder, output_gif)