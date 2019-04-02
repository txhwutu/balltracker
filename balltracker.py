import cv2, random, os
import numpy as np
from sklearn.cluster import KMeans
from time import sleep

font = cv2.FONT_HERSHEY_SIMPLEX


class ImgQueue:
  def __init__(self, maxsize=1):
    self.items = []
    self.maxsize = maxsize
    self.mid = maxsize // 2
    self.base = [-5, -4, -3, 3, 4, 5]

  def isEmpty(self):
    return self.items == []

  def isFull(self):
    return self.size() == self.maxsize

  def enqueue(self, item):
    if self.isFull():
      self.dequeue()
    self.items.insert(0, item)

  def dequeue(self):
    return self.items.pop()

  def size(self):
    return len(self.items)

  def fgdifferential(self):
    diff = np.abs(self.items[self.mid].astype(np.int) - self.items[self.mid + self.base[0]].astype(np.int)).astype(
      np.uint8)
    diff = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
    _, diff = cv2.threshold(diff, 20, 255, cv2.THRESH_BINARY)
    foreground = np.copy(diff)
    for i in self.base[1:]:
      diff = np.abs(self.items[self.mid].astype(np.int) - self.items[self.mid + i].astype(np.int)).astype(np.uint8)
      diff = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
      _, diff = cv2.threshold(diff, 20, 255, cv2.THRESH_BINARY)
      foreground = cv2.bitwise_and(foreground, diff)
    return foreground


last1 = np.array([583, 336])
last2 = np.array([582, 318])
t = 40

def cluster(kmeans, n, circles, est):
  kmeans.fit(circles)
  # for circle in circles:
  #   cv2.circle(img, (candidate[0], candidate[1]), 3, (0, 0, 255), -1)
  clt = [[] for q in range(n)]
  for p in range(len(kmeans.labels_)):
    clt[kmeans.labels_[p]].append(circles[p])
  clt = sorted(clt, key=lambda x: np.linalg.norm(np.mean(np.array(x), axis=0) - est))
  ball = sorted(clt[0], key=lambda x: np.linalg.norm(x - est))


def findball(que, n, kernel, kmeans, k):
  global last1, last2, t
  img = np.copy(que.items[que.mid])
  img_diff = que.fgdifferential()
  img_diff = cv2.morphologyEx(img_diff, cv2.MORPH_CLOSE, kernel)
  circles = cv2.HoughCircles(img_diff, cv2.HOUGH_GRADIENT, 1, 10, param1=1, param2=2, minRadius=1, maxRadius=5)
  circles = circles[0, :, :-1].astype(np.int)
  circles = np.array([pos for pos in circles if pos[1] > 95])
  kmeans.fit(circles)
  # for circle in circles:
  #   cv2.circle(img, (candidate[0], candidate[1]), 3, (0, 0, 255), -1)
  candidates = [[] for q in range(n)]
  for p in range(len(kmeans.labels_)):
    candidates[kmeans.labels_[p]].append(circles[p])
  est = last2 - last1 + last2 + 5
  r = np.linalg.norm(last1 - last2) + t
  cds = [c for c in candidates if np.linalg.norm(np.mean(np.array(c), axis=0) - last2) <= r]

  if len(cds) == 0:
    cv2.circle(img, (est[0], est[1]), 3, (0, 0, 255), -1)
    last1, last2 = last2, est
  else:
    one = [c[0] for c in cds if len(c) == 1]
    one = sorted(one, key=lambda x: np.linalg.norm(x - last2))
    if len(one) > 0:
      cv2.circle(img, (one[0][0], one[0][1]), 3, (0, 0, 255), -1)
      last1, last2 = last2, one[0]
      t = 40
    else:
      t = 200
      cds = sorted(cds, key=lambda x: min([np.linalg.norm(p - last2) for p in x]))
      cds = sorted(cds[0], key=lambda x: np.linalg.norm(x - last2))
      m = len(cds) // 2
      cv2.circle(img, (cds[0][0], cds[0][1]), 3, (0, 0, 255), -1)
      last1, last2 = last2, cds[0]
  cv2.putText(img, 'frame%d' % k, (10, 50), font, 1, (255, 255, 0), 2)
  cv2.imshow('tracking', img)


if __name__ == '__main__':
  que = ImgQueue(11)
  n = 8
  kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (10, 10))
  colors = [(random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)) for i in range(n)]
  kmeans = KMeans(n_clusters=n, random_state=0)
  frames = os.listdir('img')
  k = 5
  for f in range(len(frames)):
    img = cv2.imread('img/%s' % frames[f])
    que.enqueue(img)
    if que.isFull():
      print('processing frame %d' % k)
      findball(que, n, kernel, kmeans, k)
      k += 1

    if cv2.waitKey(50) == 27:
      break
  cv2.destroyAllWindows()
