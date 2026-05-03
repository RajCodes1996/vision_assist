import cv2
import easyocr
import numpy as np

def _resize_for_ocr(image, min_width=1200):
    height, width = image.shape[:2]
    if width >= min_width:
        return image
    scale = min_width / float(width)
    new_size = (int(width * scale), int(height * scale))
    return cv2.resize(image, new_size, interpolation=cv2.INTER_LANCZOS4)

def test_pipeline():
    # create a mock image with text
    img = np.zeros((400, 800, 3), dtype=np.uint8)
    img[:] = 255
    cv2.putText(img, "MACHINE", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 2, (0,0,0), 3)
    cv2.putText(img, "LEARNING", (50, 200), cv2.FONT_HERSHEY_SIMPLEX, 2, (0,0,0), 3)
    
    # add noise
    noise = np.random.randint(0, 50, (400, 800, 3), dtype=np.uint8)
    img = cv2.add(img, noise)

    base = _resize_for_ocr(img)
    gray = cv2.cvtColor(base, cv2.COLOR_BGR2GRAY)
    denoised = cv2.bilateralFilter(gray, 9, 75, 75)
    
    cv2.imwrite("test_base.jpg", base)
    cv2.imwrite("test_gray.jpg", gray)
    cv2.imwrite("test_denoised.jpg", denoised)
    print("Done generating test images.")

test_pipeline()
