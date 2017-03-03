import numpy as np
import cv2 as cv
from os.path import isfile,join
from os import listdir


hog = cv.HOGDescriptor('hog-properties.xml')

dirname = 'img-example'
Names=[]
for files in listdir(dirname):
	if(files.endswith('.bmp')):
		Names.append(dirname+'/'+files)
	Names.sort()

# for name in Names:
# 	imgraw = cv.imread(name,0)
# 	imgvector = imgraw.reshape(imgraw.shape[0]*imgraw.shape[1])

# 	mean, eigenvector = cv.PCACompute(imgraw,mean=np.array([]))
# 	#hogvalue = hog.compute(imgraw)
# 	eigenvector=eigenvector.reshape(eigenvector.shape[0]*eigenvector.shape[1])
# 	eigenvector = np.append(eigenvector,mean,axis=0)	
# 	print(mean.shape,eigenvector.shape)
# 	#print(hogvalue.shape)

for name in Names:
	hogvalue = hog.compute(cv.imread(name,0))
	print(hogvalue.shape)
