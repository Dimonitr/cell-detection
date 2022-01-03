# Code is based on examples from https://realpython.com/pysimplegui-python/

# Additional info about PySimpleGUI
# https://pysimplegui.readthedocs.io/en/latest/cookbook/
import PySimpleGUI as sg
import cv2
import matplotlib.pyplot as plt
import numpy as np

def adjust_gamma(image, gamma):
	# build a lookup table mapping the pixel values [0, 255] to
	# their adjusted gamma values
	invGamma = 1.0 / gamma
	table = np.array([((i / 255.0) ** invGamma) * 255
		for i in np.arange(0, 256)]).astype("uint8")
	# apply gamma correction using the lookup table
	return cv2.LUT(image, table)

def main():
    sg.theme("LightGreen")

    # Define the window layout
    layout = [
        [sg.Image(filename="", key="-IMAGE-")],
        [
            sg.Button("LOG", size=(10, 1)),
            sg.Slider(
                (-10, 10),
                0,
                0.1,
                orientation="h",
                size=(40, 10),
                key="-LOG SLIDER-",
            ),
            sg.Button("GAMMA", size=(10, 1)),
            sg.Slider(
                (0, 25),
                1,
                0.1,
                orientation="h",
                size=(40, 10),
                key="-GAMMA SLIDER-",
            ),
            
        ],
        [
            sg.Button("AVERAGE", size=(10, 1)),
            sg.Slider(
                (1, 21),
                3,
                1,
                orientation="h",
                size=(40, 10),
                key="-BLUR SLIDER-",
            ),
            sg.Button("MEDIAN", size=(10, 1)),
            sg.Slider(
                (1, 21),
                3,
                1,
                orientation="h",
                size=(40, 10),
                key="-MEDIAN SLIDER-",
            ),
            
        ],
        [
            sg.Button("HSV_THS", size=(10, 1)),
            sg.Text('H mid'),
            sg.Slider(
                (0, 360),
                180,
                1,
                orientation="h",
                size=(15, 10),
                key="-HSV SLIDER Hth-",
            ),
            sg.Text('H range'),
            sg.Slider(
                (0, 255),
                50,
                1,
                orientation="h",
                size=(15, 10),
                key="-HSV SLIDER Hr-",
            ),
            sg.Text('S Low'),
            sg.Slider(
                (0, 255),
                50,
                1,
                orientation="h",
                size=(10, 10),
                key="-HSV SLIDER S LOW-",
            ),
            sg.Text('S High'),
            sg.Slider(
                (0, 255),
                55,
                1,
                orientation="h",
                size=(10, 10),
                key="-HSV SLIDER S HIGH-",
            ),
            sg.Text('V Low'),
            sg.Slider(
                (0, 255),
                50,
                1,
                orientation="h",
                size=(10, 10),
                key="-HSV SLIDER V LOW-",
            ),
            sg.Text('V High'),
            sg.Slider(
                (0, 255),
                55,
                1,
                orientation="h",
                size=(10, 10),
                key="-HSV SLIDER V HIGH-",
            ),
        ],
        [
            sg.Button("ERODE", size=(10, 1)),
            sg.Slider(
                (1, 15),
                3,
                1,
                orientation="h",
                size=(40, 10),
                key="-ERODE SLIDER-",
            ),
            sg.Button("DILATE", size=(10, 1)),
            sg.Slider(
                (1, 15),
                3,
                1,
                orientation="h",
                size=(40, 10),
                key="-DILATE SLIDER-",
            ),
            
        ],
        [sg.Button("Reset_RGB", size=(10, 1)),sg.Button("Reset_BW", size=(10, 1)),sg.Button("Histogram", size=(10, 1)),sg.Button("Exit", size=(10, 1))],
    ]

    # Create the window and show it without the plot
    window = sg.Window("GUI Example", layout, location=(800, 400))

    img = cv2.imread('test1.jpg')
    #M, N = img.shape
    bw_image = img.copy()
    img_tmp = img.copy()
    
    frame = np.concatenate((img_tmp, bw_image), axis=1)

    while True:
        event, values = window.read(timeout=200)
        if event == "Exit" or event == sg.WIN_CLOSED:
            break
        elif event == "AVERAGE":
            b_val = int(values["-BLUR SLIDER-"])
            if (b_val % 2) == 0:
                b_val = b_val+1
            img_tmp = cv2.blur(img_tmp, (b_val, b_val), )
            frame = np.concatenate((img_tmp, bw_image), axis=1)
        elif event == "HSV_THS":
            mid_H = values["-HSV SLIDER Hth-"]
            range_H = values["-HSV SLIDER Hr-"]

            low_H = mid_H - (range_H/2)
            high_H = mid_H + (range_H/2)
            low_S = values["-HSV SLIDER S LOW-"]
            high_S = values["-HSV SLIDER S HIGH-"]
            low_V = values["-HSV SLIDER V LOW-"]
            high_V = values["-HSV SLIDER V HIGH-"]

            img_HSV = cv2.cvtColor(img_tmp, cv2.COLOR_BGR2HSV)

            frame_threshold = cv2.inRange(img_HSV, (low_H, low_S, low_V), (high_H, high_S, high_V))
            ret, thresh = cv2.threshold(frame_threshold, 127, 255, 0)
            contours, hierarchy = cv2.findContours(thresh, 1, 2)

            bw_image = cv2.cvtColor(frame_threshold,cv2.COLOR_GRAY2RGB)
            bw_contours = bw_image.copy()
            cv2.drawContours(image=bw_contours, contours=contours, contourIdx=-1, color=(0, 0, 255), thickness=1,
                             lineType=cv2.LINE_AA)

            frame = np.concatenate((img_tmp, bw_contours), axis=1)
        elif event == "LOG":
            c = 255 / np.log(values["-LOG SLIDER-"] + np.max(img_tmp))
            img_tmp = c * (np.log(img_tmp + 1))
            img_tmp = np.array(img_tmp, dtype=np.uint8)
            frame = np.concatenate((img_tmp, bw_image), axis=1)
        elif event == "GAMMA":
            g_val = values["-GAMMA SLIDER-"]
            if g_val == 0:
                g_val = 0.0001
            img_tmp = adjust_gamma(img_tmp, g_val)
            frame = np.concatenate((img_tmp, bw_image), axis=1)
        elif event == "Reset_RGB":
            img_tmp = img.copy()
            frame = np.concatenate((img_tmp, bw_image), axis=1)
        elif event == "Reset_BW":
            bw_image = img.copy()
            frame = np.concatenate((img_tmp, bw_image), axis=1)
        elif event == "MEDIAN":
            b_val = int(values["-MEDIAN SLIDER-"])
            if (b_val % 2) == 0:
                b_val = b_val+1
            img_tmp = cv2.medianBlur(img_tmp, b_val, )
            frame = np.concatenate((img_tmp, bw_image), axis=1)
        elif event == "DILATE":
            kernel = np.ones((int(values["-DILATE SLIDER-"]), int(values["-DILATE SLIDER-"])), np.uint8)
            bw_image = cv2.dilate(bw_image, kernel, 1)
            gray = cv2.cvtColor(bw_image, cv2.COLOR_BGR2GRAY)
            ret, thresh = cv2.threshold(gray, 127, 255, 0)
            contours, hierarchy = cv2.findContours(thresh, 1, 2)
            bw_contours = bw_image.copy()
            cv2.drawContours(image=bw_contours, contours=contours, contourIdx=-1, color=(0, 0, 255), thickness=1,
                             lineType=cv2.LINE_AA)
            frame = np.concatenate((img_tmp, bw_contours), axis=1)
        elif event == "ERODE":
            kernel = np.ones((int(values["-ERODE SLIDER-"]), int(values["-ERODE SLIDER-"])), np.uint8)
            bw_image = cv2.erode(bw_image, kernel, 1)
            gray = cv2.cvtColor(bw_image, cv2.COLOR_BGR2GRAY)
            ret, thresh = cv2.threshold(gray, 127, 255, 0)
            contours, hierarchy = cv2.findContours(thresh, 1, 2)
            bw_contours = bw_image.copy()
            cv2.drawContours(image=bw_contours, contours=contours, contourIdx=-1, color=(0, 0, 255), thickness=1,
                             lineType=cv2.LINE_AA)
            frame = np.concatenate((img_tmp, bw_contours), axis=1)
        elif event == "Histogram":
            gray = cv2.cvtColor(bw_image, cv2.COLOR_BGR2GRAY)
            ret, thresh = cv2.threshold(gray, 127, 255, 0)
            contours, hierarchy = cv2.findContours(thresh, 1, 2)
            areas = []
            for i in range(len(contours)):
                areas.append(cv2.contourArea(contours[i]))
            hist,bins = np.histogram(areas,5,[0,50])
            bins = ["0-10", "10-20", "20-30", "30-40", "40-50"]
            plt.stem(bins,hist)
            plt.xlabel("area (pixels)")
            plt.ylabel("number of cells")
            plt.show()

        imgbytes = cv2.imencode(".png", frame)[1].tobytes()
        window["-IMAGE-"].update(data=imgbytes)


    window.close()

main()