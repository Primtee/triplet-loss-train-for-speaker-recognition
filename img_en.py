import cv2


def en(cur_img):
    count_all = cur_img.shape[0] * cur_img.shape[1]
    count_255 = 0
    for cur_lines in cur_img:
        for cur_line in cur_lines:
            r, g, b = cur_line[:]
            if (g >= 0) and (g <= 0):
                if (b >= 0) and (b <= 0):
                    # if (b >= 120) and (b <= 128) or (b >= 240) and (b <= 255):
                    count_255 += 1
                    if float((count_255 * 1.0) / (1.0 + count_all * 1.0)) >= 0.9:  # 蓝色太多
                        return False
    return True


def good_img(img):
    # 每30个像素切分一次，转换成数组，好切片
    for margin in range(0, 200, 100):
        begin = img[:, margin: margin + 100]
        end = img[:, margin + 100: margin + 200]
        begin_mormal = en(begin)
        end_mormal = en(end)
        print(begin_mormal, end_mormal)
        if begin_mormal:  # 前一段正常
            if not end_mormal:  # 后一段不正常
                print('begin best')
                img[:, margin + 100: margin + 200] = img[:, margin: margin + 100]
        if end_mormal:  # 前一段正常
            if not begin_mormal:  # 后一段不正常
                print('end best')
                img[:, margin: margin + 100] = img[:, margin + 100: margin + 200]
    return img


def best_img(img):
    # 每30个像素切分一次，转换成数组，好切片
    begin = 150
    end = begin
    while begin >= 100:
        begin_seg = img[:, begin - 50: begin]
        begin_seg_pre = img[:, begin - 100: begin - 50]
        end_seg = img[:, end: end + 50]
        end_seg_later = img[:, end + 50: end + 1000]
        begin_mormal = en(begin_seg)
        begin_mormal_pre = en(begin_seg_pre)
        end_mormal = en(end_seg)
        end_mormal_later = en(end_seg_later)
        if begin_mormal:  # 前一段正常
            if not begin_mormal_pre:  # 后一段不正常
                img[:, begin - 100: begin - 50] = img[:, begin - 50: begin]
        if begin_mormal_pre:  # 前一段正常
            if not begin_mormal:  # 后一段不正常
                img[:, begin - 50: begin] = img[:, begin - 100: begin - 50]
        # 后半段
        if end_mormal:  # 前一段正常
            if not end_mormal_later:  # 后一段不正常
                img[:, end + 50: end + 100] = img[:, end: end + 50]
        if begin_mormal_pre:  # 前一段正常
            if not begin_mormal:  # 后一段不正常
                img[:, end: end + 30] = img[:, end + 50: end + 1000]
        begin -= 50
        end += 50
    return img


# imgs = cv2.imread('oooooo_20180904225313.png')
# imgs = cv2.resize(imgs, (300, 512))
# good_img(imgs)