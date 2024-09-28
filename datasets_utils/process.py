#from slice import slice_video
#from pathlib import Path as path
from datasets_utils.config import datasset_config
import pandas as pd
from pose.estimation import camera_pose_estimation
from calculate import calculate_mouse_movement
#import base64
from tqdm import tqdm
import os
import pyarrow.parquet as pq
import pyarrow as pa
#from fastparquet import write
#from fastparquet import ParquetFile


def image_to_binary(image_path):
    with open(image_path, 'rb') as image_file:
        binary_data = image_file.read()
    return binary_data

'''
def binary_to_base64(binary_data):
    base64_encoded = base64.b64encode(binary_data).decode('utf-8')
    return base64_encoded
'''

def clamp_value(value):
    if value > 1:
        return 1
    elif value < -1:
        return -1
    else:
        return 0

def process_slices():
    images_list = list(datasset_config.image_out_dir.iterdir())
    if len(images_list) <=10:
        print('No enough images to process')
        for image in images_list:
            image.unlink()
        return
    
    if len(images_list)>=3000:
        print('Too many images to process')
        for image in images_list:
            image.unlink()
        return

    if os.path.getsize(datasset_config.csv_file) >= 10737418240 :
        print("too large size csv")
        for image in images_list:
            image.unlink()
        os._exit()

    for i in tqdm(range(10,len(images_list)),desc="Processing"):
        pose = camera_pose_estimation(images_list[i-1],images_list[i])
        #print(pose)
        R = pose[:3,:3]
        dx, dy =calculate_mouse_movement(R)
        #print(f"鼠标相对位移: dx = {dx:.4f}, dy = {dy:.4f}")
        tx= pose[0,-1]
        ty = pose[1,-1]
        action=f'tx:{clamp_value(tx)},ty:{clamp_value(ty)},dx:{dx:.4f},dy:{dy:.4f}'
        new_line = {'info':datasset_config.current_processing_bvid,'keyword':datasset_config.current_processing_keyword, 'action':action,'previous_frame_1':image_to_binary(images_list[i-10]),'previous_frame_2':image_to_binary(images_list[i-9]),'previous_frame_3':image_to_binary(images_list[i-8]),'previous_frame_4':image_to_binary(images_list[i-7]),'previous_frame_5':image_to_binary(images_list[i-6]),'previous_frame_6':image_to_binary(images_list[i-5]),'previous_frame_7':image_to_binary(images_list[i-4]),'previous_frame_8':image_to_binary(images_list[i-3]),'previous_frame_9':image_to_binary(images_list[i-2]),'previous_frame_10':image_to_binary(images_list[i-1]),
        'current_frame':image_to_binary(images_list[i])}
        new_df = pd.DataFrame([new_line])
        new_df.to_csv(datasset_config.csv_file, mode='a', header=False, index=False
        )
    print(f'{datasset_config.current_processing_bvid} is done')
    for image in images_list:
        image.unlink()

if __name__ == "__main__":
    process_slices()
