import cv2
import numpy as np

image_name = 'watermark.png'

# Load your image
image = cv2.imread('../images/' + image_name)
# image = cv2.resize(image, (800, 600))
mask = np.zeros(image.shape[:2], dtype=np.uint8)  
drawing = False
brush_size = 10

# Mouse callback function
def draw(event, x, y, flags, param):
    global drawing
    if event == cv2.EVENT_LBUTTONDOWN:
        drawing = True
        cv2.circle(mask, (x, y), brush_size, 255, -1)
    elif event == cv2.EVENT_MOUSEMOVE:
        if drawing:
            cv2.circle(mask, (x, y), brush_size, 255, -1)
    elif event == cv2.EVENT_LBUTTONUP:
        drawing = False

cv2.namedWindow('Draw Mask')
cv2.setMouseCallback('Draw Mask', draw)

while True:
    overlay = cv2.addWeighted(image, 0.7, cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR), 0.3, 0)
    cv2.imshow('Draw Mask', overlay)

    # Adjust brush size with keys
    key = cv2.waitKey(1) & 0xFF
    if key == ord('='):  # Increase brush size
        brush_size += 2
    elif key == ord('-'):  # Decrease brush size
        brush_size = max(3, brush_size - 2)
    elif key == 27:  
        break

# Save mask
extension = image_name[-4:]
cv2.imwrite('../images/' + image_name[:-4] + '_mask' + extension, np.where(mask > 127, 0, 255))
cv2.destroyAllWindows()
