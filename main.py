import dlib, cv2
import os

detector = dlib.get_frontal_face_detector()
facerec = dlib.face_recognition_model_v1('models/dlib_face_recognition_resnet_model_v1.dat')
img_paths = 'img/'
file_list = os.listdir(img_paths)


def find_faces(img):
    DetectorObj = detector(img, 1)
    if len(DetectorObj) == 0:
        return [((0, 0), (0, 0))]
    else:
        Boundary = []
        for k, d in enumerate(DetectorObj):
            rect = ((d.left(), d.top()), (d.right(), d.bottom()))
            Boundary.append(rect)
            break
    return Boundary


fileIdx = 0
for img_name in file_list:
    try:
        img_bgr = cv2.imread(img_paths + img_name)
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        rects = find_faces(img_rgb)
        if rects[0][0][1] < rects[0][1][1] and rects[0][0][0] < rects[0][1][0]:
            cropped_img = img_rgb[rects[0][0][1]:rects[0][1][1], rects[0][0][0]:rects[0][1][0]]
            try:
                cropped_img_bgr = cv2.cvtColor(cropped_img, cv2.COLOR_RGB2BGR)
                if cropped_img_bgr is not None:
                    cv2.imwrite('cropped_img/' + img_name, cropped_img_bgr)
                    print(img_name + ': SUCCESS.', '(', fileIdx, '/', len(file_list) - 1, ')')
            except Exception:
                print(img_name + ': FAIL.', '(', fileIdx, '/', len(file_list) - 1, ')')
    except Exception:
        continue
    fileIdx = fileIdx + 1
