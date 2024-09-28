from you_get import common as you_get


def download_bilibili_with_bvid(bvid, out_dir):
    url = f'https://www.bilibili.com/video/{bvid}'
    you_get.any_download(url, output_dir=out_dir, merge=True,title=bvid)


if __name__ == '__main__':
    bvid = 'BV1Y54y1h7oM'
    download_bilibili_with_bvid(bvid, out_dir=r"C:\Users\F.F.Chopin\Downloads")

    