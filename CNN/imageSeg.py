import cv2
import numpy as np
import cv2 as cv
import cellPredictor

img = cv2.imread('test_input.tif', 0)
img_color = cv2.imread('test_input.tif', cv2.IMREAD_COLOR)
mser = cv2.MSER_create()
vis = img.copy()
regions, bboxes = mser.detectRegions(img)

hulls = [cv.convexHull(p.reshape(-1, 1, 2)) for p in regions]
cv.polylines(vis, hulls, 1, (0, 255, 0))

mask = np.zeros((img.shape[0], img.shape[1], 1), dtype=np.uint8)
mask = cv2.dilate(mask, np.ones((150, 150), np.uint8))
for contour in hulls:
    cv2.drawContours(mask, [contour], -1, (255, 255, 255), -1)

    text_only = cv2.bitwise_and(img, img, mask=mask)

for i, contour in enumerate(hulls):
    x,y,w,h = cv2.boundingRect(contour)
    print(x,y,w,h);
    newBox = cv2.resize(img_color[y:y + h, x:x + w], (64, 64))
    cell = cellPredictor.Cell(newBox, "cellClassifier.h5")
    # print("Image %i" % i)
    cell.predict()
    cv2.imwrite('{}.png'.format(i), img_color[y:y+h,x:x+w])

cv2.imshow('img', vis)
cv2.waitKey(0)
