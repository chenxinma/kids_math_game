# /usr/bin/env python
# -*- coding: utf-8 -*-
import cv2
import tensorflow as tf
import numpy as np
import math

url = 'http://192.168.2.46:4747/video'
capture = cv2.VideoCapture(url)
model = tf.keras.models.load_model('./models/mnist.h5')
SZ = 28


def deskew(img):
    m = cv2.moments(img)
    if abs(m['mu02']) < 1e-2:
        return img.copy()
    skew = m['mu11'] / m['mu02']
    M = np.float32([[1, skew, -0.5 * SZ * skew], [0, 1, 0]])
    img = cv2.warpAffine(img,
                         M, (SZ, SZ),
                         flags=cv2.WARP_INVERSE_MAP | cv2.INTER_LINEAR)
    return img


def detect(frame):
    img = frame.copy()  # 显示用图片
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  #处理用灰度图片
    gray = cv2.GaussianBlur(gray, (5, 5), 0)

    bin = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
                                cv2.THRESH_BINARY_INV, 31, 10)
    bin = cv2.medianBlur(bin, 3)

    # Threshold the image
    ret, im_th = cv2.threshold(gray, 90, 255, cv2.THRESH_BINARY_INV)

    # Find contours in the image
    ctrs, heirs = cv2.findContours(im_th.copy(), cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)

    try:
        heirs = heirs[0]
    except:
        heirs = []

    bin_set = []
    positions = []
    for cnt, heir in zip(ctrs, heirs):
        _, _, _, outer_i = heir
        if outer_i >= 0:
            continue
        x, y, w, h = cv2.boundingRect(cnt)
        if not (16 <= h <= 128 and w <= 1.2 * h):
            continue

        pad = max(h - w, 0)
        x, w = x - (pad // 2), w + pad

        # Draw the rectangles
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 3)

        bin_roi = bin[y:, x:][:h, :w]

        m = bin_roi != 0
        if not 0.1 < m.mean() < 0.4:
            continue

        s = 1.5 * float(h) / SZ
        m = cv2.moments(bin_roi)
        c1 = np.float32([m['m10'], m['m01']]) / m['m00']
        c0 = np.float32([SZ / 2, SZ / 2])
        t = c1 - s * c0
        A = np.zeros((2, 3), np.float32)
        A[:, :2] = np.eye(2) * s
        A[:, 2] = t
        bin_norm = cv2.warpAffine(bin_roi,
                                  A, (SZ, SZ),
                                  flags=cv2.WARP_INVERSE_MAP
                                  | cv2.INTER_LINEAR)
        bin_norm = deskew(bin_norm)

        if x + w + SZ < frame.shape[1] and y + SZ < frame.shape[0]:
            positions.append((x, y))
            bin_set.append(np.copy(bin_norm))

    if len(bin_set) > 0:
        bin_norms = np.array(bin_set)
        predictions = model.predict(bin_norms)
        predictions = [np.argmax(p) for p in predictions]
        ds = list(zip(positions, predictions))
        ds.sort(key=lambda x: x[0][0]*x[0][1])

        # print(list(map(lambda d: d[1], ds[:2])))

        for pt, p in zip(positions, predictions):
            cv2.putText(img, '%d' % p,
                        pt, cv2.FONT_HERSHEY_PLAIN, 2.0,
                        (255, 255, 0), 3)

    return img


while (capture.isOpened()):
    # 获取一帧
    ret, frame = capture.read()

    cv2.imshow('frame', detect(frame))
    if cv2.waitKey(1) == ord('q'):
        break

# When everything done, release the capture
capture.release()
cv2.destroyAllWindows()