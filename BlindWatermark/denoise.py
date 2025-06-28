import cv2

def remove_noise(img, k):
	def calculate_noise_count(img_obj, w, h):
		count = 0
		width, height = img_obj.shape
		for _w_ in [w - 1, w, w + 1]:
			for _h_ in [h - 1, h, h + 1]:
				if _w_ > width - 1:
					continue
				if _h_ > height - 1:
					continue
				if _w_ == w and _h_ == h:
					continue
				if img_obj[_w_, _h_] < 230:  # 二值化的图片设置为255
					count += 1
		return count

	# 灰度
	w, h = img.shape
	for _w in range(w):
		for _h in range(h):
			if _w == 0 or _h == 0:
				img[_w, _h] = 255
				continue
			# 计算邻域pixel值小于255的个数
			pixel = img[_w, _h]
			if pixel == 255:
				continue
			
			if calculate_noise_count(img, _w, _h) < k:
				img[_w, _h] = 255
				
	return img


if __name__ == '__main__':
	pass
	