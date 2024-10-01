import ffmpeg
#from search import search_bilibili
from download import download_bilibili_with_bvid
from datasets_config import datasset_config
import os



def slice_video(bvid):
    with open(datasset_config.txt_path, 'r') as file:
        processed_ids = set(line.strip() for line in file)
        if bvid in processed_ids:
            print("This bvid has been processed")
            #in_file.unlink()
            return
    with open(datasset_config.txt_path, 'a') as f:
        f.write(f'{bvid}\n')
        f.flush()  # 显式刷新缓冲区
    ret = download_bilibili_with_bvid(bvid, out_dir=datasset_config.video_out_dir)
    in_file = list(datasset_config.video_out_dir.iterdir())[0]
    out_file = os.path.join(datasset_config.image_out_dir, '%04d.jpg')
    datasset_config.current_processing_bvid = bvid
    '''
    with open(datasset_config.txt_path, 'r') as file:
            processed_ids = set(line.strip() for line in file)
            if bvid in processed_ids:
                print("This bvid has been processed")
                in_file.unlink()
                return
    '''
    if ret == True:
        if in_file.exists():
            ffmpeg.input(in_file).filter('fps', fps=10).output(out_file).run()

    if in_file.exists() and in_file.is_file():
        in_file.unlink()

if __name__ == '__main__':
    bvid = 'BV1JmtJeDE9a'
    #bvid = 'BV1uyvTeSE8HBV1iS411P7rj'
    slice_video(bvid)
    print('done')
    #print(list(datasset_config.video_out_dir.iterdir())[0])
    #print(os.path.join(datasset_config.image_out_dir, f'{bvid}_%04d.jpg'))
    
