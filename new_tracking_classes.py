import cv2 as cv
import numpy as np
import math
import random
import os
import json
import numpy

#1433 na 1075 - 0.7
#1024 na 768 - 0.5

ll_d = 768

# минимальное расстояние

min_rast = 16

# читает
DIR = r'D:\\Microfluidics\\Data\\Photo\\21.06.24\\AAA_oil_temed_susp_2\\'
DIR_2 = r'D:\Microfluidics\Data\Video\AA_oil_06.23\end.avi'
# сохраняет
path = 'D://Microfluidics//Data//Velosity_drops_profiles//21.06.23//ends'

if not os.path.isdir(path):
    os.mkdir(path)

os.chdir(path)

conc = False
photo = True
video = False


def randcol():
    r = random.randint(0, 256)
    b = random.randint(0, 256)
    g = random.randint(0, 256)
    return r, b, g


def rescaleFrame(frame, scale=0.5):
    # Images, Video and Live Video
    width = int(frame.shape[1] * scale)
    height = int(frame.shape[0] * scale)

    dimentions = (width, height)
    # print(dimentions)
    return cv.resize(frame, dimentions, interpolation=cv.INTER_AREA)


def frame_processing(frame):
    frame_resized = rescaleFrame(frame)
    gray = cv.cvtColor(frame_resized, cv.COLOR_RGB2GRAY)
    blur_gray = cv.GaussianBlur(gray, (1, 1), cv.BORDER_DEFAULT)

    return blur_gray, frame_resized


def photo_analysis(DIR):
    drops_dict = {}
    drops_conc = {}
    old_drops = {}
    p = []

    def deleting_drops():
        nonlocal drops_dict
        new_drops = drops_dict.copy()
        if photo:
            if num_frame % 10 == 0:
                for items in drops_dict:
                    if drops_dict[items][0][-1][-1] + 10 < num_frame:
                        old_drops[items] = drops_dict[items]
                        del new_drops[items]
            drops_dict = new_drops.copy()
        else:
            if nummm % 10 == 0:
                for items in drops_dict:
                    if drops_dict[items][0][-1][-1] + 10 < nummm:
                        old_drops[items] = drops_dict[items]
                        del new_drops[items]
            drops_dict = new_drops.copy()
        print(len(drops_dict))

    def search_drops(central_point):
        nonlocal drops_dict
        if not drops_dict:
            return False
        # print(central_point[0], type(central_point), drops_dict)
        min_rast_2 = numpy.inf
        our_items = None
        for items in drops_dict:
            raven = math.sqrt((central_point[0] - drops_dict[items][0][-1][0]) ** 2 + (
                    central_point[1] - drops_dict[items][0][-1][1]) ** 2)
            # print(type(drops_dict[items]), drops_dict[items])
            if raven <= min_rast_2:
                min_rast_2 = raven
                # print(items)
                our_items = items
        if min_rast_2 < min_rast:
            cv.circle(frame_resized, central_point[:2], central_point[2], drops_dict[our_items][-1].colors, thickness=2)
            cv.putText(frame_resized, 'ID#{}'.format(our_items), central_point[:2], cv.FONT_HERSHEY_TRIPLEX, 0.5,
                       drops_dict[our_items][-1].colors, 1)
            if photo:
                drops_dict[our_items][-1].add((central_point[0], central_point[1], num_frame))
            else:
                drops_dict[our_items][-1].add((central_point[0], central_point[1], nummm))
            return True
        else:
            return False

    def finding_circles():
        global ll_d
        # acrylamide
        # circles = cv.HoughCircles(frame_rebuild, cv.HOUGH_GRADIENT, dp=2, minDist=30, param1=40, param2=35,
        #                           minRadius=22, maxRadius=27)
        circles = cv.HoughCircles(frame_rebuild, cv.HOUGH_GRADIENT, dp=1.5, minDist=20, param1=50, param2=33,
                                  minRadius=12, maxRadius=18)
        # circles = cv.HoughCircles(frame_rebuild, cv.HOUGH_GRADIENT, dp=1, minDist=15, param1=50, param2=20,
        #                           minRadius=12, maxRadius=19)
        # pegda
        # circles = cv.HoughCircles(frame_rebuild, cv.HOUGH_GRADIENT, dp=2, minDist=25, param1=30, param2=30,
        #                            minRadius=17, maxRadius=21)

        def concent(cir):
            global drops_conc, ll_d
            drops_conc[num_frame] = []
            ss = 12
            hh = 0
            for i in range(ss):
                ff = True
                for (x, y, r) in cir:
                    if (y >= ll_d * i / ss) and (y <= ll_d * (i + 1) / ss):
                        hh += 1
                # y_num.append((ll_d / dd * (i+1/2)))
                drops_conc[num_frame].append(hh)
                hh = 0

        if circles is not None:

            s = 20
            h = 0
            circles = np.round(circles[0, :]).astype("int")

            if conc:
                concent(circles)

            for (x, y, r) in circles:
                h = 0
                if search_drops((x, y, r)):
                    pass
                else:
                    name = Drops.name
                    p.append(Drops())
                    drops_dict[name] = (p[-1].track, p[-1])
                    if photo:
                        drops_dict[name][-1].add((x, y, num_frame))
                    else:
                       drops_dict[name][-1].add((x, y, int(cv.CAP_PROP_FRAME_COUNT)))
                    cv.circle(frame_resized, (x, y), r, (100, 0, 0), thickness=2)
    if photo:
        for num_frame in range(len(os.listdir(DIR)) - 1):
            path_img = os.path.join(DIR, str(num_frame) + '.png')
            frame = cv.imread(path_img)
            frame_rebuild, frame_resized = frame_processing(frame)
            finding_circles()
            deleting_drops()
            cv.putText(frame_resized, str(num_frame), (15, 15), cv.FONT_HERSHEY_SIMPLEX, 0.3, (0, 0, 0))
            cv.imshow("Video", frame_resized)
            if cv.waitKey(30) & 0xFF == ord("q"):
                break
    elif video:
        capture = cv.VideoCapture(DIR_2)
        nummm = 0
        while True:
            isTrue, frame = capture.read()
            frame_rebuild, frame_resized = frame_processing(frame)
            finding_circles()
            deleting_drops()
            cv.putText(frame_resized, str(nummm), (15, 15), cv.FONT_HERSHEY_SIMPLEX, 0.3, (0, 0, 0))
            cv.imshow("Video", frame_resized)
            nummm += 1
            if cv.waitKey(30) & 0xFF == ord("q"):
                break
        capture.release()
        cv.destroyAllWindows()

    return old_drops | drops_dict, drops_conc


def convert(o):
    if isinstance(o, numpy.int32): return int(o)
    raise TypeError


class Drops:
    name = 0

    def __init__(self):
        # super().__init__()
        self.name = Drops.name
        self.colors = (randcol())
        Drops.name += 1
        self.track = []

    def add(self, v):
        self.track.append(v)


drops_conc = {}


for w in os.listdir(DIR):

    print(w)
    N_DIR = os.path.join(DIR + str(w))
    data = {key: val[0] for key, val in photo_analysis(N_DIR)[0].items()}
    cv.destroyAllWindows()

    test = 'AAA_oil_susp' + str(w) + '.json'
    with open(test, 'w') as f:
        json.dump(data, f, indent=4, sort_keys=True, default=convert)

    if conc:
        name_2 = 'conc' + str(w) + '.json'
        with open(name_2, 'w') as vf:
            json.dump(drops_conc, vf, indent=4, sort_keys=True, default=convert)
        drops_conc.clear()
