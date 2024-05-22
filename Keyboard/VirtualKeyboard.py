import cv2
from HandTrackingModule import HandDetector
from time import time
import numpy as np
import cvzone
from pynput.keyboard import Controller

cap = cv2.VideoCapture(0)
cap.set(3, 1280)
cap.set(4, 720)

detector = HandDetector(detectionCon=0.8)
keys = [["Q", "W", "E", "R", "T", "Y", "U", "I", "O", "P"],
        ["A", "S", "D", "F", "G", "H", "J", "K", "L", ";"],
        ["Z", "X", "C", "V", "B", "N", "M", ",", ".", "/"],
        [" ", "<--"]]

finalText = ""

keyboard = Controller()

isClicking = False
# Cursor blinking control
cursorVisible = True
blinkStartTime = time()
blinkInterval = 0.3  # Cursor blink interval in seconds

def drawAll(img, buttonList):
    imgNew = np.zeros_like(img, np.uint8)
    for button in buttonList:
        x, y = button.pos
        cvzone.cornerRect(imgNew, (button.pos[0], button.pos[1], button.size[0], button.size[1]),
                          20, rt=0, colorC=(255, 255, 0))
        cv2.rectangle(imgNew, button.pos, (x + button.size[0], y + button.size[1]),
                      (80, 80, 80), cv2.FILLED)
        cv2.putText(imgNew, button.text, (x + 40, y + 60),
                    cv2.FONT_HERSHEY_PLAIN, 2, (255, 255, 255), 3)

    out = img.copy()
    alpha = 0.5
    mask = imgNew.astype(bool)
    out[mask] = cv2.addWeighted(img, alpha, imgNew, 1 - alpha, 0)[mask]
    return out

class Button():
    def __init__(self, pos, text, size=[85, 85]):
        self.pos = pos
        self.size = size
        self.text = text

buttonList = []
for i in range(len(keys)):
    for j, key in enumerate(keys[i]):
        if key == " ":
            buttonList.append(Button([300, 350], key, size=[500, 85]))
        elif key == "<--":
            buttonList.append(Button([1050, 50], key, size=[85 * 2 , 85]))
        else:
            buttonList.append(Button([100 * j + 50, 100 * i + 50], key))

while True:
    success, img = cap.read()
    img = cv2.flip(img, 1)  # 1 for horizontal flip
    img = detector.findHands(img)
    lmList = detector.findPosition(img)
    img = drawAll(img, buttonList)

    if lmList:
        for button in buttonList:
            x, y = button.pos
            w, h = button.size

            if x < lmList[8][1] < x + w and y < lmList[8][2] < y + h:
                cv2.rectangle(img, (x - 5, y - 5), (x + w + 5, y + h + 5), (200, 200, 200), cv2.FILLED)
                cv2.putText(img, button.text, (x + 20, y + 65),
                            cv2.FONT_HERSHEY_PLAIN, 4, (255, 255, 255), 4)
                long, _, _ = detector.findDistance(8, 12, img, draw=True)

                # When clicked
                if long < 30:
                    if not isClicking:
                        if button.text == "<--":
                            keyboard.press('\b')
                            finalText = finalText[:-1]
                        else:
                            keyboard.press(button.text)
                            finalText += button.text
                        cv2.rectangle(img, button.pos, (x + w, y + h), (0, 255, 0), cv2.FILLED)
                        cv2.putText(img, button.text, (x + 20, y + 65),
                                    cv2.FONT_HERSHEY_PLAIN, 4, (255, 255, 255), 4) 
                        isClicking = True
                else:
                    isClicking = False

    #Output text
    cv2.rectangle(img, (50, 455), (1200, 510), (80, 80, 80), cv2.FILLED)
    cv2.putText(img, finalText, (60, 500),
                cv2.FONT_HERSHEY_PLAIN, 3, (255, 255, 255), 5)
    
    # Draw the blinking cursor
    if cursorVisible:
        textWidth, _ = cv2.getTextSize(finalText, cv2.FONT_HERSHEY_PLAIN, 3, 5)[0]
        cursorX = 60 + textWidth
        cv2.line(img, (cursorX, 465), (cursorX, 500), (255, 255, 255), 2)

    # Toggle cursor visibility
    if time() - blinkStartTime >= blinkInterval:
        cursorVisible = not cursorVisible
        blinkStartTime = time()

    cv2.imshow("Image", img)
    cv2.waitKey(1)
