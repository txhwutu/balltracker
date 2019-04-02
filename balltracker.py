import cv2, random, os
import numpy as np
from sklearn.cluster import KMeans


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


def findball(que, n, kernel, k):
  img = np.copy(que.items[que.mid])
  img_diff = que.fgdifferential()
  img_diff = cv2.morphologyEx(img_diff, cv2.MORPH_CLOSE, kernel)
  circles = cv2.HoughCircles(img_diff, cv2.HOUGH_GRADIENT, 1, 10, param1=1, param2=2, minRadius=1, maxRadius=5)
  circles = circles[0, :, :-1].astype(np.uint)
  candidates = [[] for q in range(n)]
  if len(circles) < n:
    for c in circles:
      cv2.circle(img, (c[0], c[1]), 3, (0,0,255), -1)
    cv2.imshow('less circle %d' % k, img)
    return None
  kmeans.fit(circles)
  for p in range(len(kmeans.labels_)):
    candidates[kmeans.labels_[p]].append(circles[p])
  return candidates


if __name__ == '__main__':
  que = ImgQueue(11)
  n = 8
  kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (10, 10))
  colors = [(random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)) for i in range(n)]
  kmeans = KMeans(n_clusters=n, random_state=0)
  frames = os.listdir('img')
  k = 5
  for f in range(0, min(50000, len(frames))):
    img = cv2.imread('img/%s' % frames[f])
    que.enqueue(img)
    if que.isFull():
      findball(que, n, kernel, k)
      k += 1
      print(k)
    # terminate
    if cv2.waitKey(50) == 27:
      break
  cv2.destroyAllWindows()
