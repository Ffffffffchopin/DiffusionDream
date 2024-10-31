#from slice import slice_video
#from pathlib import Path as path
from datasets_config import datasset_config
import pandas as pd

from calculate import calculate_mouse_movement
#import base64
from tqdm import tqdm
import os
import pyarrow.parquet as pq
import pyarrow as pa
#from fastparquet import write
#from fastparquet import ParquetFile
import time
#from bypy import ByPy
import sys

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)),'pose'))
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)),'pose','third_party','LoFTR'))
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)),'pose','etc','feature_matching_baselines'))
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)),'pose','third_party','prior_ransac'))

from pose.estimation import camera_pose_estimation

def create_unique_filename( extension='parquet'):

    # 获取当前时间
    current_time = time.localtime()
    
    # 格式化时间为 YYYYMMDD_HHMMSS 格式
    formatted_time = time.strftime('%Y%m%d_%H%M%S', current_time)
    
    # 构建文件名
    filename = f"{formatted_time}.{extension}"
    
    return filename


def image_to_binary(image_path):
    with open(image_path, 'rb') as image_file:
        binary_data = image_file.read()
    return binary_data

'''
def binary_to_base64(binary_data):
    base64_encoded = base64.b64encode(binary_data).decode('utf-8')
    return base64_encoded
'''

'''
def clamp_value(value):
    if value > 0.5:
        return 1
    elif value < -0.5:
        return -1
    else:
        return 0
'''

def process_slices():
    images_list = list(datasset_config.image_out_dir.iterdir())
    if len(images_list) <=10:
        print('No enough images to process')
        for image in images_list:
            image.unlink()
        return
    
    if len(images_list)>=10000:
        print('Too many images to process')
        for image in images_list:
            image.unlink()
        return


    writer=None
    for i in tqdm(range(10,len(images_list)),desc="Processing"):
        orginal_directory = os.getcwd()
        os.chdir(os.path.join(os.path.dirname(os.path.abspath(__file__)),'pose')) 
        pose = camera_pose_estimation(images_list[i-1],images_list[i])
        os.chdir(orginal_directory)
        #print(pose)
        R = pose[:3,:3]
        dx, dy =calculate_mouse_movement(R)
        #print(f"鼠标相对位移: dx = {dx:.4f}, dy = {dy:.4f}")
        tx= pose[0,-1]
        ty = pose[1,-1]
        tz = pose[2,-1]
        action=f'tx:{tx},ty:{ty},tz:{tz},dx:{dx:.4f},dy:{dy:.4f}'
        new_line = {'info':datasset_config.current_processing_bvid,'keyword':datasset_config.current_processing_keyword, 'action':action,'current_frame':image_to_binary(images_list[i]),'previous_frame_1':image_to_binary(images_list[i-1]),'previous_frame_2':image_to_binary(images_list[i-2]),'previous_frame_3':image_to_binary(images_list[i-3]),'previous_frame_4':image_to_binary(images_list[i-4]),'previous_frame_5':image_to_binary(images_list[i-5]),'previous_frame_6':image_to_binary(images_list[i-6]),'previous_frame_7':image_to_binary(images_list[i-7]),'previous_frame_8':image_to_binary(images_list[i-8]),'previous_frame_9':image_to_binary(images_list[i-9]),'previous_frame_10':image_to_binary(images_list[i-10])
        }
        new_df = pd.DataFrame([new_line])
        #new_df.to_csv(datasset_config.csv_file, mode='a', header=False, index=False)
        table = pa.Table.from_pandas(new_df)
        parquet_files = list(datasset_config.parquet_path.iterdir())
        if writer is None:
            filename=create_unique_filename()
            #pq.write_table(table,os.path.join(datasset_config.parquet_path,filename))
            writer = pq.ParquetWriter(os.path.join(datasset_config.parquet_path,filename), table.schema)
            writer.write_table(table=table)
            parquet_files.append(os.path.join(datasset_config.parquet_path,filename))
            
        else:
            writer.write_table(table=table)
            #writer.close()
        if os.path.getsize(parquet_files[0]) >= 10737418240:
            print("too large size parquet")
            #os.remove(parquet_files[0])
            for image in images_list:
                image.unlink()
            writer.close()
            sys.exit()
            
            
        
    print(f'{datasset_config.current_processing_bvid} is done')

    for image in images_list:
        image.unlink()
    writer.close()

if __name__ == "__main__":
    process_slices()
    #print(sys.path)
