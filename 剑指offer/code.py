import numpy as np


# -*- coding:utf-8 -*-
class Solution:
  def minNumberInRotateArray(self, rotateArray):
    # write code here

    def MinOrder(arr, L, R):
      result = arr[L]
      for val in arr:
        if val < result:
          result = val
      return result

    def process(arr, L, R):
      mid = L
      while (arr[L] >= arr[R]):
        if L == R - 1:
          mid = R
          break

        mid = (L + R) >> 1
        if arr[L] == arr[mid] and arr[L] == arr[R]:
          return MinOrder(arr, L, R)

        if arr[L] > arr[mid]:
          R = mid
        else:
          L = mid

      return arr[mid]

    return process(rotateArray, 0, len(rotateArray) - 1)

# arr_1 = [1,2,3,4,5,6]
# arr_2 = [3,4,5,6,7,8,2,3,3,3,3,3]
# s = Solution()
# print(s.minNumberInRotateArray(arr_1))
# print(s.minNumberInRotateArray(arr_2))

def MatrixMultiply(M1, M2):
  '''
  return M1 * M2
  :param M1: List
  :param M2: List
  :return result: List
  '''
  if (not M1) or (not M2):
    return []

  assert len(M1[0]) == len(M2)
  nrow, nmid, ncol = len(M1), len(M1[0]), len(M2[0])
  result = [[0 for _ in range(ncol)] for _ in range(nrow)]

  for i in range(nrow):
    for j in range(ncol):
      for k in range(nmid):
        result[i][j] += M1[i][k] * M2[k][j]
  return result

# M = [[1, 1], [1, 0]]
def FastMatrixPow(M, n):
  result = [[1, 0], [0, 1]]
  base = M
  while n>0:
    if n&0x1:
      result = MatrixMultiply(result, base)
    base = MatrixMultiply(base, base)
    n >>= 1
  return result

M1 = [[1,2,3],[2,3,4]]
M2 = [[1,1,0],[2,2,2],[3,3,3]]
# print(MatrixMultiply(M1, M2))

M = [[1,1],[1,0]]
print(FastMatrixPow(M, 3))
print(np.matmul(M, M))