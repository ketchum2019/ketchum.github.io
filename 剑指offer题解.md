# 剑指offer题解
## 68. 最低公共祖先

```
public TreeNode lowestCommonAncestor(TreeNode root, TreeNode p, TreeNode q) {
    if (root == null || root == p || root == q)
        return root;
    TreeNode left = lowestCommonAncestor(root.left, p, q);
    TreeNode right = lowestCommonAncestor(root.right, p, q);
    return left == null ? right : right == null ? left : root;
}
```

## 67. 把字符串转换成整数 ##

> 题目：将一个字符串转换成一个整数(实现Integer.valueOf(string)的功能，但是string不符合数字要求时返回0)，要求不能使用字符串转换整数的库函数。 数值为0或者字符串不是一个合法的数值则返回0。

```
public class Solution {
    public int StrToInt(String str) {
        if (str == null || str.length() == 0) {
            return 0;
        }
        int num = 0, index = 0;
        boolean minus = false;
        if (str.charAt(0) == '+') {
            index++;
        } else if (str.charAt(0) == '-') {
            minus = true;
            index++;
        }
        while (index < str.length()) {
            char digit = str.charAt(index++);
            if (digit >= '0' && digit <= '9') {
                num = num * 10 + digit - '0';
            } else {
                return 0;
            }
        }
        return minus ? -num : num;
    }
}
```

## 66. 构建乘积数组

> 题目：给定一个数组A[0,1,...,n-1]，请构建一个数组B[0,1,...,n-1]，其中B中的元素B[i]=A[0]×A[1]×...×A[i-1]×A[i+1]×...×A[n-1]。不能使用除法。

```
public class Solution {
    public int[] multiply(int[] A) {
        if (A == null || A.length < 2) {
            return A;
        }
        int[] B = new int[A.length];
        B[0] = 1;
        for (int i = 1; i < A.length; i++) {
            B[i] = B[i - 1] * A[i - 1];
        }
        for (int i = A.length - 2, temp = 1; i >= 0; i--) {
            temp *= A[i + 1];
            B[i] *= temp;
        }
        return B;
    }
}
```

## 65. 不用加减乘除做加法

> 写一个函数，求两个整数之和，要求在函数体内不得使用+、-、*、/四则运算符号。

```
public class Solution {
    public int Add(int num1,int num2) {
        int sum = num1;
        while(num2 !=0){
            sum = num1 ^ num2;
            num2 = (num1 & num2) << 1;
            num1 = sum;
        }
        return sum;
    }
}
```

## 64. 求 1+2+3+...+n

> 要求不能使用乘除法、for、while、if、else、switch、case 等关键字及条件判断语句 A ? B : C。

使用递归解法最重要的是指定返回条件，但是本题无法直接使用 if 语句来指定返回条件。

条件与 && 具有短路原则，即在第一个条件语句为 false 的情况下不会去执行第二个条件语句。利用这一特性，将递归的返回条件取非然后作为 && 的第一个条件语句，递归的主体转换为第二个条件语句，那么当递归的返回条件为 true 的情况下就不会执行递归的主体部分，递归返回。

本题的递归返回条件为 n <= 0，取非后就是 n > 0；递归的主体部分为 sum += Sum_Solution(n - 1)，转换为条件语句后就是 (sum += Sum_Solution(n - 1)) > 0。

```
public int Sum_Solution(int n) {
    int sum = n;
    boolean b = (n > 0) && ((sum += Sum_Solution(n - 1)) > 0);
    return sum;
}
```

## 62. 圆圈中最后剩下的数字 ##

>  题目：0,1,···,n-1这n个数字排成一个圆圈，从数字0开始，每次从这个圆圈里删除第m个数字。求出这个圆圈里剩下的最后一个数字。

```
//环形链表模拟圆圈
import java.util.LinkedList;

public class Solution {
    public int LastRemaining_Solution(int n, int m) {
        if (n < 1 || m < 1) {
            return -1;
        }
        LinkedList<Integer> numbers = new LinkedList<>();
        for (int i = 0; i < n; i++) {
            numbers.add(i);
        }
        int current = 0;
        while (numbers.size() > 1) {
            current = (current + m - 1) % numbers.size();
            numbers.remove(current);
        }
        return numbers.getFirst();
    }
}
```

```
//递推公式：f(n,m) = [f(n-1,m)+m]%n，n>1
public class Solution {
    public int LastRemaining_Solution(int n, int m) {
        if (n < 1 || m < 1) {
            return -1;
        }
        int last = 0;
        for (int i = 2; i <= n; i++) {
            last = (last + m) % i;
        }
        return last;
    }
}
```

## 61. 扑克牌中的顺子 ##

> 题目：从扑克牌中随机抽5张牌，判断是不是一个顺子，即这5张牌是不是连续的。2~10为数字本身，A为1，J为11，Q为12，K为13，而大、小王可以看成任意数字。为了方便起见，你可以认为大小王是0。
>
> https://www.nowcoder.com/practice/762836f4d43d43ca9deb273b3de8e1f4

```
import java.util.Arrays;

public class Solution {
    public boolean isContinuous(int[] numbers) {
        if (numbers == null || numbers.length != 5) {
            return false;
        }
        Arrays.sort(numbers);
        int zero = 0;
        // 统计数组中0的个数
        for (int i = 0; i < numbers.length && numbers[i] == 0; i++) {
            zero++;
        }
        int gap = 0;
        // 统计数组中的间隔数目
        for (int i = zero + 1; i < numbers.length; i++) {
            // 两个数相等，有对子，不可能是顺子
            if (numbers[i - 1] == numbers[i]) {
                return false;
            }
            gap += numbers[i] - numbers[i - 1] - 1;
        }
        return gap <= zero;
    }
}
```

## 60. n个骰子的点数 ##

> 题目：把n个骰子扔在地上，所有骰子朝上一面的点数之和为s。输入n，打印出s的所有可能的值出现的概率。

```

```



## 59. 队列的最大值 ##

> 题目一：滑动窗口的最大值。
>
> 给定一个数组和滑动窗口的大小，找出所有滑动窗口里数值的最大值。例如，如果输入数组{2,3,4,2,6,2,5,1}及滑动窗口的大小3，那么一共存在6个滑动窗口，他们的最大值分别为{4,4,6,6,6,5}； 针对数组{2,3,4,2,6,2,5,1}的滑动窗口有以下6个： {[2,3,4],2,6,2,5,1}， {2,[3,4,2],6,2,5,1}， {2,3,[4,2,6],2,5,1}， {2,3,4,[2,6,2],5,1}， {2,3,4,2,[6,2,5],1}， {2,3,4,2,6,[2,5,1]}。

```
import java.util.ArrayList;
import java.util.LinkedList;

public class Solution {
    public ArrayList<Integer> maxInWindows(int[] num, int size) {
        ArrayList<Integer> result = new ArrayList<>();
        if (num == null || num.length == 0 || size < 1 || size > num.length) {
            return result;
        }
        LinkedList<Integer> list = new LinkedList<>();
        for (int i = 0; i < size; i++) {
            while (!list.isEmpty() && num[list.getLast()] <= num[i]) {
                list.removeLast();
            }
            list.add(i);
        }
        for (int i = size; i < num.length; i++) {
            result.add(num[list.getFirst()]);
            while (!list.isEmpty() && num[list.getLast()] <= num[i]) {
                list.removeLast();
            }
            if (!list.isEmpty() && i - list.getFirst() >= size) {
                list.removeFirst();
            }
            list.add(i);
        }
        result.add(num[list.getFirst()]);
        return result;
    }
}
```

## 58. 翻转字符串

> 题目一：翻转单词顺序。
>
> 输入一个英文句子，翻转句子中单词的顺序，但单词内字符的顺序不变。为简单起见，标点符号和普通字母一样处理。例如输入字符串"I am a student."，则输出"student. a am I"

```
public class Solution {
    public String ReverseSentence(String str) {
        if(str ==null ||str.length() ==0) return str;
        String[] str1 = str.split(" ");
        if (str1==null ||str1.length ==0) return str;
        StringBuilder s = new StringBuilder();
        for(int i = str1.length-1; i>=0; i--){
            if(i != 0){
                s.append(str1[i]).append(' ');
            }else{
                s.append(str1[i]);
            }

        }
        return s.toString();
    }
}
```

```
//解法二
public class Solution {
    public String ReverseSentence(String str) {
        if (str == null || str.length() == 0) {
            return str;
        }
        char[] data = str.toCharArray();
        // 翻转整个句子
        reverse(data, 0, data.length - 1);
        // 翻转句子中的每个单词
        int start = 0, end = 0;
        while (start < data.length) {
            if (data[start] == ' ') {
                start++;
                end++;
            } else if (end == data.length || data[end] == ' ') {
                reverse(data, start, end - 1);
                start = ++end;
            } else {
                end++;
            }
        }
        return new String(data);
    }
    
    private void reverse(char[] data, int start, int end) {
        while (start < end) {
            char temp = data[start];
            data[start] = data[end];
            data[end] = temp;
            start++;
            end--;
        }
    }
}
```



> 题目二：左旋转字符串
>
> 字符串的左旋转操作是把字符串前面的若干个字符转移到字符串的尾部。请定义一个函数实现字符串左旋转操作的功能。比如，输入字符串"abcdefg"和数字2，该函数将返回左旋转两位得到的结果"cdefgab"。

```
public class Solution {
    public String LeftRotateString(String str, int n) {
        if (str == null || str.length() == 0 || n < 1 || n >= str.length()) {
            return str;
        }
        char[] data = str.toCharArray();
        // 翻转字符串的前面n个字符
        reverse(data, 0, n - 1);
        // 翻转字符串的后面部分
        reverse(data, n, data.length - 1);
        // 翻转整个字符串
        reverse(data, 0, data.length - 1);
        return new String(data);
    }
    
    private void reverse(char[] data, int start, int end) {
        while (start < end) {
            char temp = data[start];
            data[start] = data[end];
            data[end] = temp;
            start++;
            end--;
        }
    }
}
```

## 57. 和为s的数字

> 题目一：和为s的两个数字。
>
> 输入一个递增排序的数组和一个数字S，在数组中查找两个数，使得他们的和正好是S，如果有多对数字的和等于S，则输出任意一对即可

## 56. 数组中数字出现的次数 ##

> 题目一：数组中只出现一次的两个数字。
>
> 一个整型数组里除了两个数字之外，其他的数字都出现了两次。请写程序找出这两个只出现一次的数字。要求时间复杂度是O(n)，空间复杂度是O(1)。

两个不相等的元素在位级表示上必定会有一位存在不同，将数组的所有元素异或得到的结果为不存在重复的两个元素异或的结果。

diff &= -diff 得到出 diff 最右侧不为 0 的位，也就是不存在重复的两个元素在位级表示上最右侧不同的那一位，利用这一位就可以将两个元素区分开来。

```
public void FindNumsAppearOnce(int[] nums, int num1[], int num2[]) {
    int diff = 0;
    for (int num : nums)
        diff ^= num;
    diff &= -diff;
    for (int num : nums) {
        if ((num & diff) == 0)
            num1[0] ^= num;
        else
            num2[0] ^= num;
    }
}
```

## 55. 二叉树的深度

> 题目一：二叉树的深度
>
> 输入一棵二叉树，求该树的深度。从根结点到叶结点依次经过的结点（含根、叶结点）形成树的一条路径，最长路径的长度为树的深度。

```
public class Solution {
    public int TreeDepth(TreeNode root) {
        if (root == null) {
            return 0;
        }
        return Math.max(TreeDepth(root.left), TreeDepth(root.right)) + 1;
    }
}
```

> 题目二：平衡二叉树
>
> 输入一棵二叉树，判断该二叉树是否是平衡二叉树。如果某二叉树中任意节点的左、右子树的深度相差不超过1，那么它就是一棵平衡二叉树。

```
public class Solution {
    private int mDepth;
    
    public boolean IsBalanced_Solution(TreeNode root) {
        if (root == null) {
            mDepth = 0;
            return true;
        }
        if (!IsBalanced_Solution(root.left)) {
            return false;
        }
        int left = mDepth;
        if (!IsBalanced_Solution(root.right)) {
            return false;
        }
        int right = mDepth;
        int diff = left - right;
        if (diff <= 1 && diff >= -1) {
            mDepth = 1 + (left > right ? left : right);
            return true;
        } else {
            return false;
        }
    }
}
```

## 54. 二叉搜索树的第k大节点

> 题目：给定一棵二叉搜索树，请找出其中的第k小的结点。例如，（5，3，7，2，4，6，8）中，按结点数值大小顺序第三小结点的值为4。
>
> [https://www.nowcoder.com/practice/ef068f602dde4d28aab2b210e859150a]

中序遍历可以找出二叉搜索树的第k大节点

```
private TreeNode ret;
private int cnt = 0;

public TreeNode KthNode(TreeNode pRoot, int k) {
    inOrder(pRoot, k);
    return ret;
}

private void inOrder(TreeNode root, int k) {
    if (root == null || cnt >= k)
        return;
    inOrder(root.left, k);
    cnt++;
    if (cnt == k)
        ret = root;
    inOrder(root.right, k);
}
```

## 55. 在排序数组中查找数字

> 题目一：数字在排序数组中出现的次数。
>
> 统计一个数字在排序数组中出现的次数。例如，输入排序数组{1, 2, 3, 3, 3, 3, 4, 5}和数字3，由于3在这个数组中出现了4次，因此输出4。

```
public int GetNumberOfK(int[] nums, int K) {
    int first = binarySearch(nums, K);
    int last = binarySearch(nums, K + 1);
    return (first == nums.length || nums[first] != K) ? 0 : last - first;
}

private int binarySearch(int[] nums, int K) {
    int l = 0, h = nums.length;
    while (l < h) {
        int m = l + (h - l) / 2;
        if (nums[m] >= K)
            h = m;
        else
            l = m + 1;
    }
    return l;
}
```

> 题目二：0~n-1中缺失的数字
>
> 一个长度为n-1的递增排序数组中的所有数字都是唯一的，并且每个数字都在范围0 ~ n-1之内。在范围0 ~ n-1内的n个数字中有且只有一个数字不在该数组中，请找出这个数字
>
> 解题思路：二分查找：找到第一个元素和下标不等的数字

```
public int getMissingNumber(int[] numbers) {
    if (numbers == null || numbers.length == 0) {
        return -1;
    }
    int left = 0, right = numbers.length - 1;
    while (left <= right) {
        int mid = (left + right) >> 1;
        if (numbers[mid] != mid) {
            if (mid == 0 || numbers[mid - 1] == mid - 1) {
                return mid;
            }
            right = mid - 1;
        } else {
            left = mid + 1;
        }
    }
    if (left == numbers.length) {
        return left;
    }
    // 无效的输入，比如数组不是按要求排序的，
    // 或者有数字不在0~n-1范围之内
    return -1;
}
```

## 52. 两个链表的第一个公共节点

```
public class Solution {
    public ListNode FindFirstCommonNode(ListNode pHead1, ListNode pHead2) {
        if (pHead1 == null || pHead2 == null) {
            return null;
        }
        ListNode p1 = pHead1, p2 = pHead2;
        // 得到两个链表的长度
        int len1 = 0, len2 = 0;
        while (p1 != null) {
            len1++;
            p1 = p1.next;
        }
        while (p2 != null) {
            len2++;
            p2 = p2.next;
        }
        int delta = len2 - len1;
        p1 = pHead1;
        p2 = pHead2;
        // 先在长链表上走几步，再同时在两个链表上遍历
        if (delta < 0) {
            while (delta != 0) {
                p1 = p1.next;
                delta++;
            }
        } else {
            while (delta != 0) {
                p2 = p2.next;
                delta--;
            }
        }
        while (p1 != p2) {
            p1 = p1.next;
            p2 = p2.next;
        }
        // 得到第一个公共节点
        return p1;
    }
}
```

## 51. 数组中的逆序对

> 题目：在数组中的两个数字，如果前面一个数字大于后面的数字，则这两个数字组成一个逆序对。输入一个数组，求出这个数组中的逆序对的总数P。并将P对1000000007取模的结果输出，即输出P%1000000007。例如，在数组{7, 5, 6, 4}中，一共存在5个逆序对，分别是(7, 6)、(7, 5)、(7, 4)、(6, 4)和(5, 4)。
>
> 解题思路：归并排序

```java
public class Solution {
    public int InversePairs(int[] array) {
        if (array == null || array.length == 0) {
            return 0;
        }
        int[] copy = new int[array.length];
        for (int i = 0; i < array.length; i++) {
            copy[i] = array[i];
        }
        return inverse(array, copy, 0, array.length - 1);
    }
    
    private int inverse(int[] array, int[] copy, int start, int end) {
        if (start == end) {
            return 0;
        }
        int mid = (start + end) / 2;
        int left = inverse(copy, array, start, mid);
        int right = inverse(copy, array, mid + 1, end);
        int leftIndex = mid, rightIndex = end, copyIndex = end, count = 0;
        while (leftIndex >= start && rightIndex > mid) {
            if (array[leftIndex] > array[rightIndex]) {
                copy[copyIndex--] = array[leftIndex--];
                count += rightIndex - mid;
                if (count >= 1000000007) {
                    count %= 1000000007;
                }
            } else {
                copy[copyIndex--] = array[rightIndex--];
            }
        }
        while (leftIndex >= start) {
            copy[copyIndex--] = array[leftIndex--];
        }
        while (rightIndex > mid) {
            copy[copyIndex--] = array[rightIndex--];
        }
        return (left + right + count) % 1000000007;
    }
}
```

## 50. 第一个只出现一次的字符

> 题目一：字符串中第一个只出现一次的字符。
>
> 在一个字符串(0<=字符串长度<=10000，全部由字母组成)中找到第一个只出现一次的字符,并返回它的位置, 如果没有则返回 -1（需要区分大小写）。

解决方案：哈希表

```java
public class Solution {
    public int FirstNotRepeatingChar(String str) {
        if (str == null || str.length() == 0) {
            return -1;
        }
        int[] counts = new int[256];
        for (int i = 0; i < str.length(); i++) {
            char c = str.charAt(i);
            counts[c]++;
        }
        for (int i = 0; i < str.length(); i++) {
            if (counts[str.charAt(i)] == 1) {
                return i;
            }
        }
        return -1;
    }
}
```

## 49. 丑数

> 题目：把只包含质因子2、3和5的数称作丑数（Ugly Number）。例如6、8都是丑数，但14不是，因为它包含质因子7。 习惯上我们把1当做是第一个丑数。求按从小到大的顺序的第N个丑数。

```java
public class Solution {
    public int GetUglyNumber_Solution(int index) {
        if (index <= 0) {
            return 0;
        }
        int[] ugly = new int[index];
        ugly[0] = 1;
        for (int i = 1, multiply2 = 0, multiply3 = 0, multiply5 = 0; i < index; i++) {
            ugly[i] = Math.min(ugly[multiply2] * 2, Math.min(ugly[multiply3] * 3, ugly[multiply5] * 5));
            while (ugly[multiply2] * 2 <= ugly[i]) {
                multiply2++;
            }
            while (ugly[multiply3] * 3 <= ugly[i]) {
                multiply3++;
            }
            while (ugly[multiply5] * 5 <= ugly[i]) {
                multiply5++;
            }
        }
        return ugly[index - 1];
    }
}
```

## 48. 最长不含重复字符的子字符串

> 题目：请从字符串中找出一个最长的不包含重复字符的子字符串，计算该最长子字符串的长度。假设字符串中只包含'a'~'z'的字符。例如，在字符串"arabcacfr"中，最长的不含重复字符的子字符串是"acfr"，长度为4。

```java
class Solution {
    public int lengthOfLongestSubstring(String s) {
        Map<Character, Integer> dic = new HashMap<>();
        int res = 0, tmp = 0;
        for(int j = 0; j < s.length(); j++) {
            int i = dic.getOrDefault(s.charAt(j), -1); // 获取索引 i
            dic.put(s.charAt(j), j); // 更新哈希表
            tmp = tmp < j - i ? tmp + 1 : j - i; // dp[j - 1] -> dp[j]
            res = Math.max(res, tmp); // max(dp[j - 1], dp[j])
        }
        return res;
    }
}
```

## 47. 礼物的最大价值

> 题目：在一个m×n的棋盘的每一格都放有一个礼物，每个礼物都有一定的价值（价值大于0）。你可以从棋盘的左上角开始拿格子里的礼物，并每次向右或者向下移动一格，直到到达棋盘的右下角。给定一个棋盘及其上面的礼物，请计算你最多能拿到多少价值的礼物？

```java
class Solution {
    public int maxValue(int[][] grid) {
        int m = grid.length, n = grid[0].length;
        for(int i = 0; i < m; i++) {
            for(int j = 0; j < n; j++) {
                if(i == 0 && j == 0) continue;
                if(i == 0) grid[i][j] += grid[i][j - 1] ;
                else if(j == 0) grid[i][j] += grid[i - 1][j];
                else grid[i][j] += Math.max(grid[i][j - 1], grid[i - 1][j]);
            }
        }
        return grid[m - 1][n - 1];
    }
}
```

## 46. 把数字翻译成字符串

> 题目：给定一个数字，我们按照如下规则把它翻译为字符串：0翻译成“a”，1翻译成“b”，……，11翻译成“l”，……，25翻译成“z”。一个数字可能有多个翻译。例如，12258有5种不同的翻译，分别是“bccfi”、“bwfi”、“bczi”、“mcfi”和“mzi”。请编程实现一个函数，用来计算一个数字有多少种不同的翻译方法。
>
> https://leetcode-cn.com/problems/ba-shu-zi-fan-yi-cheng-zi-fu-chuan-lcof/solution/mian-shi-ti-46-ba-shu-zi-fan-yi-cheng-zi-fu-chua-6/

```java
class Solution {
    public int translateNum(int num) {
        String s = String.valueOf(num);
        int a = 1, b = 1;
        for(int i = 2; i <= s.length(); i++) {
            String tmp = s.substring(i - 2, i);
            int c = tmp.compareTo("10") >= 0 && tmp.compareTo("25") <= 0 ? a + b : a;
            b = a;
            a = c;
        }
        return a;
    }
}
```

```java
class Solution {
    public int translateNum(int num) {
        int a = 1, b = 1, x, y = num % 10;
        while(num != 0) {
            num /= 10;
            x = num % 10;
            int tmp = 10 * x + y;
            int c = (tmp >= 10 && tmp <= 25) ? a + b : a;
            b = a;
            a = c;
            y = x;
        }
        return a;
    }
}
```

## 45. 把数组排成最小的数

> 题目：输入一个正整数数组，把数组里所有数字拼接起来排成一个数，打印能拼接出的所有数字中最小的一个。例如输入数组{3，32，321}，则打印出这三个数字能排成的最小数字为321323。
>
> https://leetcode-cn.com/problems/ba-shu-zu-pai-cheng-zui-xiao-de-shu-lcof/solution/mian-shi-ti-45-ba-shu-zu-pai-cheng-zui-xiao-de-s-4/

```java
class Solution {
    public String minNumber(int[] nums) {
        String[] strs = new String[nums.length];
        for(int i = 0; i < nums.length; i++) 
            strs[i] = String.valueOf(nums[i]);
        Arrays.sort(strs, (x, y) -> (x + y).compareTo(y + x));
        StringBuilder res = new StringBuilder();
        for(String s : strs)
            res.append(s);
        return res.toString();
    }
}
```

## 44. 数字序列中某一位的数字

> 题目：数字以0123456789101112131415···的格式序列化到一个字符序列中。在这个序列中，第5位（从0开始计数）是5，第13位是1，第19位是4，等等。请写一个函数，求任意第n位对应的数字。
>
> https://leetcode-cn.com/problems/shu-zi-xu-lie-zhong-mou-yi-wei-de-shu-zi-lcof/solution/mian-shi-ti-44-shu-zi-xu-lie-zhong-mou-yi-wei-de-6/

```java
class Solution {
    public int findNthDigit(int n) {
        int digit = 1;
        long start = 1;
        long count = 9;
        while (n > count) { // 1.
            n -= count;
            digit += 1;
            start *= 10;
            count = digit * start * 9;
        }
        long num = start + (n - 1) / digit; // 2.
        return Long.toString(num).charAt((n - 1) % digit) - '0'; // 3.
    }
}
```

## 43. 1~n整数中1出现的次数

> 题目：输入一个整数n，求1 ~ n这n个整数的十进制表示中1出现的次数。例如，输入12，1 ~ 12这些整数中包含1的数字有1、10、11和12，1一共出现了5次。

