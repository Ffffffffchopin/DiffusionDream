from search import search_bilibili
from slice import slice_video
from process import process_slices
from datasets_utils.config import datasset_config

def pipeline(keyword,max_page):
    bvids = search_bilibili(keyword, max_page)
    print(f'processing {len(bvids)}')
    for bvid in bvids:
        slice_video(bvid)
        process_slices()

if __name__ == '__main__':
    #keywords = ['瓦罗兰特','无畏契约','CSGO','我的世界']
    keywords = ['瓦罗兰特']
    for keyword in keywords:
        datasset_config.current_processing_keyword = keyword
        max_page = 1
        pipeline(keyword, max_page)
