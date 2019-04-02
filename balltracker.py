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
    diff = np.abs(self.items[self.mid].astype(np.int) - self.items[self.mid + self.base[0]].astype(np.int)).astype(np.uint8)
    diff = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
    _, diff = cv2.threshold(diff, 20, 255, cv2.THRESH_BINARY)
    foreground = np.copy(diff)
    for i in self.base[1:]:
      diff = np.abs(self.items[self.mid].astype(np.int) - self.items[self.mid + i].astype(np.int)).astype(np.uint8)
      diff = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
      _, diff = cv2.threshold(diff, 20, 255, cv2.THRESH_BINARY)
      foreground = cv2.bitwise_and(foreground, diff)
    return foreground


que = ImgQueue(11)
kernel_dilation1 = cv2.getStructuringElement(cv2.MORPH_RECT, (70, 70))
kernel_dilation2 = cv2.getStructuringElement(cv2.MORPH_RECT, (20, 20))
kernel_close = cv2.getStructuringElement(cv2.MORPH_RECT, (10, 10))
center = np.array([640, 360])


def filter(clusters):
  lt = []
  t = []
  for i in clusters:
    mid = i[1] + i[3] // 2
    if 80 <= mid < 180 or 370 <= mid < 520:
      lt.append(i)
    else:
      t.append(i)
  if len(lt) < 2:
    return None, None
  b2l = sorted(lt, key=lambda x: x[4], reverse=True)
  players = b2l[:2]
  ball = []
  for i in range(2, len(b2l)):
    t.append(b2l[i])
  for i in t:
    if players[1][1] + players[1][3] // 2 < i[1] + i[3] // 2 < players[0][1] + players[0][3] // 2:
      if 200 < i[0] + i[2] // 2 < 1000:
        ball.append(i)
  # if len(ball) == 0:
  #   ball = [None]
  # else:
  #   ball = sorted(ball, key=lambda x: np.linalg.norm(x[:2] + x[2:4] / 2 - center))
  return players, ball

if __name__ == '__main__':
  n = 8
  colors = [(random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)) for i in range(n)]
  kmeans = KMeans(n_clusters=n, random_state=0)
  frames = os.listdir('img')
  for f in range(50, len(frames)):
    img = cv2.imread('img/%s' % frames[f])
    que.enqueue(img)
    if que.isFull():
      img_backup = np.copy(que.items[que.mid])
      img_diff = que.fgdifferential()
      cv2.imshow('debug0', img_diff)
      # img_dilation = cv2.morphologyEx(img_diff, cv2.MORPH_CLOSE, kernel_close)
      # cv2.imshow('debug1', img_dilation)
      # nlabels, labels, stats, centroids = cv2.connectedComponentsWithStats(img_dilation)
      # clusters_dilation = stats[1:, :]
      # players, balls = filter(clusters_dilation)
      # if players == None:
      #   continue
      # for player in players:
      #   # if cluster[4] > 6500:
      #   cv2.rectangle(img_backup, (player[0], player[1]), (player[0] + player[2], player[1] + player[3]), (0, 0, 255))
      # # if ball is not None:
      # for ball in balls:
      #   # if cluster[4] > 6500:
      #   cv2.rectangle(img_backup, (ball[0], ball[1]), (ball[0] + ball[2], ball[1] + ball[3]), (0, 0, 255))
      # cv2.imshow('a', img_backup)
      # if cluster[4] < 1000:
      #   img_dilation[cluster[1]:cluster[1] + cluster[3], cluster[0]:cluster[0] + cluster[2]] = 0
      # img_diff[img_dilation == 255] = 0
      # cv2.imshow('debug3', img_diff)
      circles = cv2.HoughCircles(img_diff, cv2.HOUGH_GRADIENT, 1, 10, param1=1, param2=2, minRadius=1, maxRadius=5)
      if circles is not None:
        points = []
        result = [[] for q in range(n)]
        k = 0
        circles = circles[0, :, :-1].astype(np.int)
        if len(circles) < n:
          continue
        kmeans.fit(circles)
        lables = kmeans.labels_
        for p in range(len(lables)):
          result[lables[p]].append(circles[p])
        for i in result:
          if len(i) == 1:
            for c in i:
              if 200 < c[0] < 850 and c[1] < 550:
                cv2.circle(img_backup, (c[0], c[1]), 3, (0,0,255), -1)
          k += 1
      cv2.imshow('debug', img_backup)
    # terminate
    if cv2.waitKey(50) == 27:
      break
  cap.release()
  cv2.destroyAllWindows()
