from search import search_bilibili
from slice import slice_video
from process import process_slices

def pipeline(keyword,max_page):
    bvids = search_bilibili(keyword, max_page)
    print(f'processing {len(bvids)}')
    for bvid in bvids:
        slice_video(bvid)
        process_slices()

if __name__ == '__main__':
    keywords = ['瓦罗兰特','无畏契约','CSGO','我的世界']
    for keyword in keywords:
        max_page = 10
        pipeline(keyword, max_page)
