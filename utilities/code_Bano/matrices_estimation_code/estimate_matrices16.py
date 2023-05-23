import cv2
import matplotlib
from PIL import Image
import numpy as np
import os
import PIL.Image
import os
from PIL import Image
import cv2
import numpy as np

from tqdm import tqdm



# matplotlib.use('TkAgg')
#sorted(os.listdir(os.path.join(os.getcwd(), 'final_dataset')))
# FARE UN VIDEO ALLA VOLTA
path_frames = os.path.join(os.getcwd(), '../../../final_dataset/anon016')
list_frames_names = sorted(os.listdir(os.path.join(path_frames, 'predicted_mask')))
errorThresh = 0.0001

path_matrices = os.path.join(os.getcwd(), '../../../final_dataset/anon016/output_sp_RANSAC')
if not os.path.exists(path_matrices):
	os.makedirs(path_matrices)


maxIter = 50
eccPass = True
maskFound = False

for i in range(len(list_frames_names)):

	#file list contiene coppie di frame successivi
	actualH = np.eye(3, dtype='d')
	if i > 0 :

		mapper = cv2.reg_MapperGradAffine()
		mappPyr = cv2.reg_MapperPyramid(mapper)
		mappPyr.numLev_ = 5
		mappPyr.numIterPerScale_ = 500
		map = cv2.reg_Map(actualH)

		print(os.path.join(path_frames, 'predicted_mask', list_frames_names[i-1]))
		print(os.path.join(path_frames, 'predicted_mask', list_frames_names[i]))

		fixed = np.float32(cv2.imread(os.path.join(path_frames, 'predicted_mask', list_frames_names[i]), 0)) / 255.0
		moving = np.float32(cv2.imread(os.path.join(path_frames, 'predicted_mask', list_frames_names[i + 1]), 0)) / 255.0
		iterNum = 0
		goodH = np.eye(3, dtype=np.float32)
		map = cv2.reg_Map(np.eye(3, dtype=np.float32))
		errorV = np.inf
		error = np.inf
		while iterNum < maxIter and error > errorThresh:
			iterNum += 1
			resmap = mappPyr.calculate(moving, fixed, map)
			map = cv2.reg.MapTypeCaster_toAffine(resmap)
			lintr = map.getLinTr()
			shift = map.getShift()
			actualH = np.eye(3, dtype=np.float32)
			actualH[:2, :2] = lintr
			actualH[:2, 2] = shift[:, 0]
			# actualH = map.getProjTr()
			warped = cv2.warpPerspective(moving, actualH, dsize=fixed.shape[:2])
			map = cv2.reg_Map(actualH)
			try:
				error = (np.abs(moving - warped).sum()) / np.product(warped.shape[:2])
				if error < errorV:
					errorV = error
					goodH = actualH
				else:
					actualH = goodH
				map = cv2.reg_Map(actualH)
			except (RuntimeError, TypeError, NameError, ValueError):
				pass

	name_matrix = list_frames_names[i].replace('.png', '')
	np.savetxt(os.path.join(path_matrices, name_matrix + '.txt'), actualH, delimiter=' ')
	print(actualH)