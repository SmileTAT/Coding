### 1、二维数组中的查找
#### 题目描述
在一个二维数组中（每个一维数组的长度相同），每一行都按照从左到右递增的顺序排序，每一列都按照从上到下递增的顺序排序。请完成一个函数，输入这样的一个二维数组和一个整数，判断数组中是否含有该整数。

#### 题目解析
从二维矩阵matrix左下角（右上角）开始，所在列往上递减，所在行往右递增，两者相连即有序，所在点的意义与二分查找中点意义相似，与目标值比大小后更新i，j

#### 题目代码
```
# -*- coding:utf-8 -*-
class Solution:
    # array 二维列表
    def Find(self, target, array):
        # write code here
        nrow, ncol = len(array), len(array[0])
        
        i, j = nrow-1, 0
        while(i>=0 and j<=ncol-1):
          if array[i][j]==target:
            return True
          elif array[i][j]<target:
            j += 1
          else:
            i -= 1
        return False
``` 

### 2、重建二叉树
#### 题目描述
输入某二叉树的前序遍历和中序遍历的结果，请重建出该二叉树。假设输入的前序遍历和中序遍历的结果中都不含重复的数字。例如输入前序遍历序列{1,2,4,7,3,5,6,8}和中序遍历序列{4,7,2,1,5,3,8,6}，则重建二叉树并返回。

#### 题目解析
前序+中序（后序+中序）可得后序（前序）  
从前序知根结点，再根据结点值从中序分出左子树和右子树，递归  
注意本题中无重复值
#### 题目代码
```
# -*- coding:utf-8 -*-
# class TreeNode:
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None
class Solution:
    # 返回构造的TreeNode根节点
    def reConstructBinaryTree(self, pre, tin):
        # write code here
        return self.process(pre, tin)
      
    def process(self, pre, tin):
        if not pre:
            return None
        
        root_val = pre[0]
        root = TreeNode(root_val)
        root.left = self.process(pre[1:tin.index(root_val)+1], tin[:tin.index(root_val)])
        root.right = self.process(pre[tin.index(root_val)+1:], tin[tin.index(root_val)+1:])
        return root
```

### 3、旋转数组的最小值
#### 题目描述
把一个数组最开始的若干个元素搬到数组的末尾，我们称之为数组的旋转。 输入一个非减排序的数组的一个旋转，输出旋转数组的最小元素。 例如数组{3,4,5,1,2}为{1,2,3,4,5}的一个旋转，该数组的最小值为1。 NOTE：给出的所有元素都大于0，若数组大小为0，请返回0。

#### 题目解析
二分查找变形  
若严格递增而不是非减则可做到时间复杂度O(logN)  
非减情况O(N)  
首尾相比确定是否为旋转数组,若首、中、尾相等则顺序查找O(N)，否则比较首尾和arr\[mid]更新L，R  
牛客网测试用例不全面

#### 题目代码
```
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

# test
arr_1 = [1,2,3,4,5,6]
arr_2 = [3,4,5,6,7,8,2,3,3,3,3,3]
s = Solution()
print(s.minNumberInRotateArray(arr_1))
print(s.minNumberInRotateArray(arr_2))
```

### 4、斐波那契数列
#### 题目描述
大家都知道斐波那契数列，现在要求输入一个整数n，请你输出斐波那契数列的第n项（从0开始，第0项为0）。
n<=39

#### 题目解析
1. 递归
2. 迭代
3. 矩阵快速幂 时间复杂度O(logN)
4. 公式法，逗我呢>.<

#### 题目代码
1-递归
```
666
```
2-迭代
```
666
```
3-矩阵快速幂
```
# Note: 任意矩阵乘法时间复杂度 优化方法（Google）
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

M1 = [[1,2,3],[2,3,4]]
M2 = [[1,1,0],[2,2,2],[3,3,3]]
print(MatrixMultiply(M1, M2))

 
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

```
4-公式  
$a_n = \frac{1}{\sqrt{5}}*[(\frac{1+\sqrt{5}}{2})^n-(\frac{1-\sqrt{5}}{2})^n]$

### 5、二叉树中和为某一值的路径
#### 题目描述
输入一颗二叉树的跟节点和一个整数，打印出二叉树中结点值的和为输入整数的所有路径。路径定义为从树的根结点开始往下一直到叶结点所经过的结点形成一条路径。(注意: 在返回值的list中，数组长度大的数组靠前)

#### 题目解析
深搜、回溯、深拷贝

#### 题目代码
```
# -*- coding:utf-8 -*-
# class TreeNode:
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None
import copy

class Solution:
    # 返回二维列表，内部每个列表表示找到的路径
    def __init__(self):
        self.result = []
    def FindPath(self, root, expectNumber):
        # write code here
        self.dfs([], root, expectNumber)
        return self.result

    def dfs(self, path, root, expectNumber):
        if root==None:
            pass
        elif (root.left==None) and (root.right==None) and (expectNumber==root.val):
            path.append(root.val)
            self.result.append(copy.deepcopy(path))
        else:
            path.append(root.val)
            expectNumber -= root.val
            self.dfs(copy.deepcopy(path), root.left, expectNumber)
            self.dfs(copy.deepcopy(path), root.right, expectNumber)
            
```
### 6、复杂链表复制
#### 题目描述
输入一个复杂链表（每个节点中有节点值，以及两个指针，一个指向下一个节点，另一个特殊指针指向任意一个节点），返回结果为复制后复杂链表的head。（注意，输出结果中请不要返回参数中的节点引用，否则判题程序会直接返回空）

#### 题目解析
1-根据next复制链表
2-根据random复制链表
3-拆分

#### 题目代码
```
# -*- coding:utf-8 -*-
# class RandomListNode:
#     def __init__(self, x):
#         self.label = x
#         self.next = None
#         self.random = None
class Solution:
    # 返回 RandomListNode
    def Clone(self, pHead):
        # write code here
        if pHead == None:
            return
        
        p = pHead
        while(p!=None):
            next_p = p.next
            node = RandomListNode(p.label)
            p.next = node
            node.next = next_p
            p = next_p

        p = pHead
        new_p = p.next
        
        while(p!=None):
            if p.random:
                p.next.random = p.random.next
            p = p.next.next
        
        p = pHead
        while(p.next!=None):
            next_p = p.next
            p.next = p.next.next
            p = next_p
        return new_p
```

### 7、二叉树与双向链表
#### 题目描述
输入一棵二叉搜索树，将该二叉搜索树转换成一个排序的双向链表。要求不能创建任何新的结点，只能调整树中结点指针的指向。

#### 题目解析
二叉搜索树中序遍历为有序

当前结点的left指向中序遍历前一结点，前一结点right指向当前结点

#### 题目代码
```
# -*- coding:utf-8 -*-
# class TreeNode:
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None
class Solution:
    def __init__(self):
        self.head = None
        self.prenode = None
            
    def Convert(self, pRootOfTree):
        # write code here
        self.process(pRootOfTree, None)
        return self.head
    
    def process(self, root, prenode):
        if root==None:
            return
        
        self.process(root.left, root)
        if self.prenode:
            self.prenode.right = root
            root.left = self.prenode
        else:
            self.head = root
        self.prenode = root
        self.process(root.right, root)
        
```

### 8、最小的K个数
#### 题目描述
输入n个整数，找出其中最小的K个数。例如输入4,5,1,6,2,7,3,8这8个数字，则最小的4个数字是1,2,3,4,。

#### 题目解析
Top K问题

1、维护大小为K的大根堆 O(NlogK)

2、快排 partition 平均O(N)

3、bfprt O(N), 有点难，暂时搞不定 >.<

#### 题目代码
