# background = visualization_img
import cv2



import cv2
import numpy as np

def overlay_transparent(background, overlay, x, y):

    background_width = background.shape[1]
    background_height = background.shape[0]

    if x >= background_width or y >= background_height:
        return background

    h, w = overlay.shape[0], overlay.shape[1]

    if x + w > background_width:
        w = background_width - x
        overlay = overlay[:, :w]

    if y + h > background_height:
        h = background_height - y
        overlay = overlay[:h]

    if overlay.shape[2] < 4:
        overlay = np.concatenate(
            [
                overlay,
                np.ones((overlay.shape[0], overlay.shape[1], 1), dtype = overlay.dtype) * 255
            ],
            axis = 2,
        )

    overlay_image = overlay[..., :3]
    mask = overlay[..., 3:] / 255.0

    background[y:y+h, x:x+w] = (1.0 - mask) * background[y:y+h, x:x+w] + mask * overlay_image

    return background

a = cv2.imread('download.png')
overlay = cv2.imread('left.png')
# overlay = cv2.resize(overlay,(120,12))
cv2.imshow('image',overlay)
rows,cols,channels = overlay.shape
print(rows,cols)
# overlay=cv2.addWeighted(a[0:rows, 0:0+cols],0.5,overlay,0.5,0)
# dst = cv2.addWeighted(a, 0.5, overlay, 0.7, 0)
print(a.shape)
dst =overlay_transparent(a,overlay,10,10)
cv2.imshow('image',dst)
cv2.waitKey(0)



# background[250:250+rows, 0:0+cols ] = overlay