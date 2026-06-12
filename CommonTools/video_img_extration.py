import subprocess
import os

def createPath(file_path):
    """
    create a given path if not exist and return it
    :param file_path:
    :return: file_path
    """
    if os.path.exists(file_path) is False:
        os.makedirs(file_path)
    return file_path

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


def ffmpeg_tif_to_png(tif_folder, output_folder):
    """
    将TIFF图片文件夹中的图片转换为PNG格式。

    :param tif_folder: TIFF图片文件夹路径
    :param output_folder: 输出文件夹路径
    """
    # 获取TIFF图片文件夹中的所有图片文件
    tif_files = [os.path.join(tif_folder, f) for f in os.listdir(tif_folder) if f.endswith('.tif') or f.endswith('.tiff')]
    
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

def ffmpeg_combine_mp4(input_file, output_file):
    """
    使用FFmpeg将多个视频文件合并为一个。

    参数:
    input_file (str): 输入视频文件的路径。
    output_file (str): 输出视频文件的路径。
    """
    command = [
        'ffmpeg',
        '-i', input_file,
        '-c', 'copy',
        output_file
    ]
    subprocess.run(command, check=True)

def ffmpeg_extract_video_folder(input_folder, output_folder):
    """
    提取文件夹中所有MP4视频的视频流并去除音频。

    参数:
    input_folder (str): 包含MP4视频文件的文件夹路径。
    output_folder (str): 输出视频文件（无音频）的文件夹路径。
    """
    # 获取文件夹中所有MP4文件
    mp4_files = [os.path.join(input_folder, f) for f in os.listdir(input_folder) if f.endswith('.mp4')]
    
    # 检查是否有MP4文件
    if not mp4_files:
        print("文件夹中没有MP4视频文件。")
        return
    
    # 确保输出文件夹存在
    os.makedirs(output_folder, exist_ok=True)
    
    # 遍历MP4文件并提取视频流
    for mp4_file in mp4_files:
        # 构建输出文件路径
        output_file = os.path.join(output_folder, os.path.basename(mp4_file))
        
        # 构建FFmpeg命令
        command = [
            'ffmpeg',
            '-i', mp4_file,
            '-c:v', 'copy',  # 复制视频流，不重新编码
            '-an',  # 禁用音频
            output_file
        ]
        
        # 执行FFmpeg命令
        subprocess.run(command, check=True)
        print(f"视频流已提取到: {output_file}")

def ffmpeg_combine_mp4_audio(video_path, audio_path, output_path):
    """
    将视频和音频合并，如果音频时长短于视频时长，则循环音频以匹配视频时长。

    参数:
    video_path (str): 输入视频文件的路径。
    audio_path (str): 输入音频文件的路径。
    output_path (str): 输出视频文件的路径。
    """
    # 使用FFmpeg的stream_loop参数循环音频
    command = [
        'ffmpeg',
        '-stream_loop', '-1',  # -1表示无限循环
        '-i', audio_path,     # 输入音频文件
        '-i', video_path,     # 输入视频文件
        '-c:v', 'copy',       # 复制视频流
        '-c:a', 'aac',        # 音频编码为AAC
        '-shortest',          # 当最短的流结束时停止编码
        '-map', '0:a',        # 使用第一个输入的音频流
        '-map', '1:v',        # 使用第二个输入的视频流
        output_path
    ]
    subprocess.run(command, check=True)
    print(f"视频和音频已合并: {output_path}")

def ffmpeg_extract_video(input_file, output_file):
    """
    从MP4视频中提取视频流（无音频）。

    参数:
    input_file (str): 输入视频文件的路径。
    output_file (str): 输出视频文件（无音频）的路径。
    """
    command = [
        'ffmpeg',
        '-i', input_file,
        '-c:v', 'copy',  # 复制视频流，不重新编码
        '-an',  # 禁用音频
        output_file
    ]
    subprocess.run(command, check=True)
    print(f"视频流已提取到: {output_file}")


def ffmpeg_extract_audio(input_file, output_file):
    """
    从MP4视频中提取音频流（无视频）。

    参数:
    input_file (str): 输入视频文件的路径。
    output_file (str): 输出音频文件的路径。
    """
    command = [
        'ffmpeg',
        '-i', input_file,
        '-c:a', 'libmp3lame',  # 复制音频流，不重新编码
        '-vn',  # 禁用视频
        output_file
    ]
    subprocess.run(command, check=True)
    print(f"音频流已提取到: {output_file}")

def ffmpeg_combine_mp4_folder(input_folder, output_file):
    """
    使用FFmpeg将文件夹中的多个MP4视频文件合并为一个MP4视频。

    参数:
    input_folder (str): 包含MP4视频文件的文件夹路径。
    output_file (str): 输出合并后的视频文件路径。
    """
    # 获取文件夹中所有MP4文件
    mp4_files = [os.path.join(input_folder, f) for f in os.listdir(input_folder) if f.endswith('.mp4')]
    
    # 检查是否有MP4文件
    if not mp4_files:
        print("文件夹中没有MP4视频文件。")
        return
    
    # 对文件进行排序，确保正确的拼接顺序
    mp4_files.sort()
    
    # 创建临时文件列表
    list_file = os.path.join(input_folder, 'file_list.txt')
    with open(list_file, 'w') as f:
        for mp4_file in mp4_files:
            # 使用绝对路径，避免路径问题
            abs_path = os.path.abspath(mp4_file)
            f.write(f"file '{abs_path}'\n")
    
    # 构建FFmpeg命令
    command = [
        'ffmpeg',
        '-f', 'concat',
        '-safe', '0',
        '-i', list_file,
        '-c', 'copy',
        '-an',
        output_file
    ]
    
    # 执行FFmpeg命令
    subprocess.run(command, check=True,shell=True)
    
    # 删除临时文件列表
    os.remove(list_file)
    print(f"视频已合并并保存到: {output_file}")

# 示例用法
if __name__ == '__main__':
    # video_path = os.path.abspath(r'J:\Films\201210_0701_1080P_4000K_378043122.mp4')
    # output_folder = os.path.join(os.path.dirname(video_path), 'frames')
    # #create output_folder
    # #extract_frames(video_path, output_folder)
    # print('视频帧提取完成!')
    
    # tif_folder= os.path.abspath(r'T:\FastXimages\AlO_bubble_growth')
    # png_folder = createPath(os.path.join(os.path.dirname(tif_folder), os.path.basename(tif_folder)+'_png'))
    # ffmpeg_tif_to_png( tif_folder, png_folder)
    # # png to video
    # #image_folder = os.path.abspath(r'I:\Coding\GitReposity\ImageProcessScripts\img\bubble_1d8H2_png')
    # output_video=os.path.join(png_folder , 'output_video.mp4')
    # ffmpeg_images_to_video(png_folder, output_video)
    # # png to gif
    # output_gif=os.path.join(png_folder , 'output_gif.gif')
    # ffmpeg_images_to_gif(png_folder, output_gif)
    # video=r'R:\CODEs\chuntingxue\videoplayback.mp4'
    # audio=r'R:\CODEs\chuntingxue\videoplayback.weba'
    # output_video=os.path.join(os.path.dirname(video), 'output_video.mp4')
    # ffmpeg_combine_mp4_audio(video, audio, output_video)
    # combine folder videos
    input_folder =os.path.abspath(r'H:\Ximages\VideoIn01')
    output_folder=createPath(os.path.abspath(r'H:\Ximages\VideoIn01_woAudio'))
    ffmpeg_extract_video_folder(input_folder, output_folder)
    output_file = os.path.join(os.path.dirname(input_folder), 'combined_Fullvideo04.mp4')
    ffmpeg_combine_mp4_folder(output_folder, output_file)
    # extract video audio
    video_path = os.path.abspath(r'H:\Ximages\allureGirls_dance.mp4')
    audio_file= os.path.join(os.path.dirname(video_path), 'allureGirls_dance.mp3')
    #ffmpeg_extract_audio(video_path, audio_file)
    # combine video audio
    videonew_file= os.path.join(r'H:\Ximages\combined_Fullvideo04.mp4')
    output_file = os.path.join(os.path.dirname(videonew_file), 'Allure_video03.mp4')
    ffmpeg_combine_mp4_audio(videonew_file, audio_file,  output_file)