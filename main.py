import dlib, cv2
import os, time
from pytube import YouTube

detector = dlib.get_frontal_face_detector()
facerec = dlib.face_recognition_model_v1('models/dlib_face_recognition_resnet_model_v1.dat')
img_paths = 'img/'
video_paths = 'video/'
file_list = os.listdir(img_paths)
video_list = os.listdir(video_paths)

def find_faces(img):
    DetectorObj = detector(img, 1)
    if len(DetectorObj) == 0:
        return [((0, 0), (0, 0))]
    else:
        Boundary = []
        for k, d in enumerate(DetectorObj):
            rect = ((d.left(), d.top()), (d.right(), d.bottom()))
            Boundary.append(rect)
    return Boundary


def FaceExport():
    fileIdx = 0
    for img_name in file_list:
        try:
            img_bgr = cv2.imread(img_paths + img_name)
            img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
            rects = find_faces(img_rgb)
            for rectIdx in range(len(rects)):
                if rects[rectIdx][0][1] + 150 < rects[rectIdx][1][1] \
                        and rects[rectIdx][0][0] + 150 < rects[rectIdx][1][0]:
                    cropped_img = img_rgb[rects[rectIdx][0][1]:rects[rectIdx][1][1],
                                  rects[rectIdx][0][0]:rects[rectIdx][1][0]]
                    try:
                        cropped_img_bgr = cv2.cvtColor(cropped_img, cv2.COLOR_RGB2BGR)
                        if cropped_img_bgr is not None:
                            cv2.imwrite('cropped_img/' + img_name, cropped_img_bgr)
                            print(img_name + ': SUCCESS.', '(', fileIdx, '/', len(file_list) - 1, ')')
                    except Exception:
                        print(img_name + ': FAIL.', '(', fileIdx, '/', len(file_list) - 1, ')')
                else:
                    print(img_name + ': FAIL.', '(', fileIdx, '/', len(file_list) - 1, ')')
        except Exception:
            continue
        fileIdx = fileIdx + 1


def VedieoExtraction():
    for infile in video_list:
        outfile = img_paths + infile + 'frames'
        second = 0.1
        vidcap = cv2.VideoCapture(video_paths+infile)
        fps = int(vidcap.get(cv2.CAP_PROP_FPS))
        write_frames(vidcap, outfile, fps, second)


def DownloadVideoFromYoutube(Address):
    print('Download URL: ',Address)
    print('Downloading...')
    YouTube(Address).streams.filter(file_extension='mp4').get_highest_resolution().download(output_path='video', filename='FaceSet')
    print('Download completed.')


def write_frames(vidcap, outfile, fps, second):
    success = True
    counter = 0
    NameCounter = 0
    while success:
        success, frame = vidcap.read()
        if counter > second * fps:
            cv2.imwrite('{}%d.jpg'.format(outfile) % NameCounter, frame)
            print(NameCounter, 'frame written')
            counter = 0
            NameCounter += 1
        else:
            counter += 1


if __name__ == "__main__":
    #DownloadVideoFromYoutube('https://www.youtube.com/watch?v=AvbwyGH9a3E')
    VedieoExtraction()
    time.sleep(10)
    FaceExport()
