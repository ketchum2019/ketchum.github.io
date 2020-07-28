# 剑指offer题解

## 68. 最低公共祖先

```java
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

```java
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

```java
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

```java
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

```java
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

```java
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

```java
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

```java
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

```java
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

```java
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

```java
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

```java
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

```java
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

```java
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

```java
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

```java
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

## 43. 1~n整数中1出现的次数（未理解）

> 题目：输入一个整数n，求1 ~ n这n个整数的十进制表示中1出现的次数。例如，输入12，1 ~ 12这些整数中包含1的数字有1、10、11和12，1一共出现了5次。
>
> https://leetcode.com/problems/number-of-digit-one/discuss/64381/4+-lines-O(log-n)-C++JavaPython

```java
public int NumberOf1Between1AndN_Solution(int n) {
    int cnt = 0;
    for (int m = 1; m <= n; m *= 10) {
        int a = n / m, b = n % m;
        cnt += (a + 8) / 10 * m + (a % 10 == 1 ? b + 1 : 0);
    }
    return cnt;
}
```

## 42. 连续子数组的最大和

> 题目：输入一个整型数组，数组里有正数也有负数。数组中的一个或连续多个整数组成一个子数组。求所有子数组的和的最大值。要求时间复杂度为O(n)。
>
> 例如，输入的数组为{1, -2, 3, 10, -4, 7, 2, -5}，和最大的子数组为{3, 10, -4, 7, 2}，因此输出为该子数组的和18。

```java
public class Solution {
    public int FindGreatestSumOfSubArray(int[] array) {
        if (array == null || array.length == 0) {
            return 0;
        }
        int sum = 0, max = Integer.MIN_VALUE;
        for (int num : array) {
            if (sum <= 0) {
                sum = num;
            } else {
                sum += num;
            }
            if (sum > max) {
                max = sum;
            }
        }
        return max;
    }
}
```

## 41. 数据流中的中位数

> 题目：如何得到一个数据流中的中位数？如果从数据流中读出奇数个数值，那么中位数就是所有数值排序之后位于中间的数值。如果从数据流中读出偶数个数值，那么中位数就是所有数值排序之后中间两个数的平均值。我们使用Insert()方法读取数据流，使用GetMedian()方法获取当前读取数据的中位数。

```java
/* 大顶堆，存储左半边元素 */
private PriorityQueue<Integer> left = new PriorityQueue<>((o1, o2) -> o2 - o1);
/* 小顶堆，存储右半边元素，并且右半边元素都大于左半边 */
private PriorityQueue<Integer> right = new PriorityQueue<>();
/* 当前数据流读入的元素个数 */
private int N = 0;

public void Insert(Integer val) {
    /* 插入要保证两个堆存于平衡状态 */
    if (N % 2 == 0) {
        /* N 为偶数的情况下插入到右半边。
         * 因为右半边元素都要大于左半边，但是新插入的元素不一定比左半边元素来的大，
         * 因此需要先将元素插入左半边，然后利用左半边为大顶堆的特点，取出堆顶元素即为最大元素，此时插入右半边 */
        left.add(val);
        right.add(left.poll());
    } else {
        right.add(val);
        left.add(right.poll());
    }
    N++;
}

public Double GetMedian() {
    if (N % 2 == 0)
        return (left.peek() + right.peek()) / 2.0;
    else
        return (double) right.peek();
}

```

## 41.2 字符流中第一个不重复的字符

```java
private int[] cnts = new int[256];
private Queue<Character> queue = new LinkedList<>();

public void Insert(char ch) {
    cnts[ch]++;
    queue.add(ch);
    while (!queue.isEmpty() && cnts[queue.peek()] > 1)
        queue.poll();
}

public char FirstAppearingOnce() {
    return queue.isEmpty() ? '#' : queue.peek();
}
```

## 40. 最小的K个数

> 题目：输入n个整数，找出其中最小的K个数。例如输入4,5,1,6,2,7,3,8这8个数字，则最小的4个数字是1,2,3,4。

应该使用大顶堆来维护最小堆，而不能直接创建一个小顶堆并设置一个大小，企图让小顶堆中的元素都是最小元素。

维护一个大小为 K 的最小堆过程如下：在添加一个元素之后，如果大顶堆的大小大于 K，那么需要将大顶堆的堆顶元素去除。

```java
public ArrayList<Integer> GetLeastNumbers_Solution(int[] nums, int k) {
    if (k > nums.length || k <= 0)
        return new ArrayList<>();
    PriorityQueue<Integer> maxHeap = new PriorityQueue<>((o1, o2) -> o2 - o1);
    for (int num : nums) {
        maxHeap.add(num);
        if (maxHeap.size() > k)
            maxHeap.poll();
    }
    return new ArrayList<>(maxHeap);
}
```

## 39. 数组中超过一半的数字

> 使用 cnt 来统计一个元素出现的次数，当遍历到的元素和统计元素相等时，令 cnt++，否则令 cnt--。如果前面查找了 i 个元素，且 cnt == 0，说明前 i 个元素没有 majority，或者有 majority，但是出现的次数少于 i / 2 ，因为如果多于 i / 2 的话 cnt 就一定不会为 0 。此时剩下的 n - i 个元素中，majority 的数目依然多于 (n - i) / 2，因此继续查找就能找出 majority。

```java
public int MoreThanHalfNum_Solution(int[] nums) {
    int majority = nums[0];
    for (int i = 1, cnt = 1; i < nums.length; i++) {
        cnt = nums[i] == majority ? cnt + 1 : cnt - 1;
        if (cnt == 0) {
            majority = nums[i];
            cnt = 1;
        }
    }
    int cnt = 0;
    for (int val : nums)
        if (val == majority)
            cnt++;
    return cnt > nums.length / 2 ? majority : 0;
}
```

## 38. 字符串的排列

> 题目：输入一个字符串，按字典序打印出该字符串中字符的所有排列。例如输入字符串abc，则打印出由字符a,b,c所能排列出来的所有字符串abc,acb,bac,bca,cab和cba。

```java
import java.util.*;
public class Solution {
    ArrayList<String> res =new ArrayList<>();
    public ArrayList<String> Permutation(String str) {
        if(str == null ||str.length()==0) return res;
        char[] chars = str.toCharArray();
        backtracking(chars, new boolean[chars.length],new StringBuilder(""));
        return res;
    }
    public void backtracking(char[] chars, boolean[] hasused, StringBuilder s){
        if(s.length() == chars.length) res.add(new String(s));
        for(int i=0; i<chars.length; i++){
            if(hasused[i]) continue;
            if(i!=0 && chars[i]==chars[i-1] && !hasused[i-1]) continue;
            hasused[i]=true;
            s.append(chars[i]);
            backtracking(chars, hasused,s);
            s.deleteCharAt(s.length()-1);
            hasused[i]=false;
        }
    }
}
```

## 37. 序列化二叉树

>  题目：请实现两个函数，分别用来序列化和反序列化二叉树。可以根据前序遍历的顺序来序列化二叉树。在遍历二叉树碰到null时，这些null序列化为一个特殊字符'$'。另外，节点的数值之间要用一个特殊字符','隔开。

```java
private String deserializeStr;

public String Serialize(TreeNode root) {
    if (root == null)
        return "#";
    return root.val + " " + Serialize(root.left) + " " + Serialize(root.right);
}

public TreeNode Deserialize(String str) {
    deserializeStr = str;
    return Deserialize();
}

private TreeNode Deserialize() {
    if (deserializeStr.length() == 0)
        return null;
    int index = deserializeStr.indexOf(" ");
    String node = index == -1 ? deserializeStr : deserializeStr.substring(0, index);
    deserializeStr = index == -1 ? "" : deserializeStr.substring(index + 1);
    if (node.equals("#"))
        return null;
    int val = Integer.valueOf(node);
    TreeNode t = new TreeNode(val);
    t.left = Deserialize();
    t.right = Deserialize();
    return t;
}
```

## 36. 二叉搜索树和双向链表

> 题目：输入一棵二叉搜索树，将该二叉搜索树转换成一个排序的双向链表。要求不能创建任何新的结点，只能调整树中结点指针的指向。
>
> 解决思路：中序遍历
>
> [https://cyc2018.github.io/CS-Notes/#/notes/36.%20%E4%BA%8C%E5%8F%89%E6%90%9C%E7%B4%A2%E6%A0%91%E4%B8%8E%E5%8F%8C%E5%90%91%E9%93%BE%E8%A1%A8](https://cyc2018.github.io/CS-Notes/#/notes/36. 二叉搜索树与双向链表)

```java
private TreeNode pre = null;
private TreeNode head = null;

public TreeNode Convert(TreeNode root) {
    inOrder(root);
    return head;
}

private void inOrder(TreeNode node) {
    if (node == null)
        return;
    inOrder(node.left);
    node.left = pre;
    if (pre != null)
        pre.right = node;
    pre = node;
    if (head == null)
        head = node;
    inOrder(node.right);
}
```

## 35. 复杂链表的复制

> 题目：输入一个复杂链表（每个节点中有节点值，以及两个指针，一个指向下一个节点，另一个特殊指针指向任意一个节点），返回结果为复制后复杂链表的head。
>
> 第一步，在每个节点的后面插入复制的节点。
>
> 第二步，对复制节点的 random 链接进行赋值。
>
> 第三步，拆分

```java
public RandomListNode Clone(RandomListNode pHead) {
    if (pHead == null)
        return null;
    // 插入新节点
    RandomListNode cur = pHead;
    while (cur != null) {
        RandomListNode clone = new RandomListNode(cur.label);
        clone.next = cur.next;
        cur.next = clone;
        cur = clone.next;
    }
    // 建立 random 链接
    cur = pHead;
    while (cur != null) {
        RandomListNode clone = cur.next;
        if (cur.random != null)
            clone.random = cur.random.next;
        cur = clone.next;
    }
    // 拆分
    cur = pHead;
    RandomListNode pCloneHead = pHead.next;
    while (cur.next != null) {
        RandomListNode next = cur.next;
        cur.next = next.next;
        cur = next;
    }
    return pCloneHead;
}

```

## 34. 二叉树中和为某一值的路径

> 题目：输入一颗二叉树的根节点和一个整数，打印出二叉树中结点值的和为输入整数的所有路径。路径定义为从树的根结点开始往下一直到叶结点所经过的结点形成一条路径。

```java
private ArrayList<ArrayList<Integer>> ret = new ArrayList<>();

public ArrayList<ArrayList<Integer>> FindPath(TreeNode root, int target) {
    backtracking(root, target, new ArrayList<>());
    return ret;
}

private void backtracking(TreeNode node, int target, ArrayList<Integer> path) {
    if (node == null)
        return;
    path.add(node.val);
    target -= node.val;
    if (target == 0 && node.left == null && node.right == null) {
        ret.add(new ArrayList<>(path));
    } else {
        backtracking(node.left, target, path);
        backtracking(node.right, target, path);
    }
    path.remove(path.size() - 1);
}

```

## 33. 二叉搜索树的后序遍历序列

> 题目：输入一个整数数组，判断该数组是不是某二叉搜索树的后序遍历的结果。如果是则输出Yes,否则输出No。假设输入的数组的任意两个数字都互不相同。

```java
public boolean VerifySquenceOfBST(int[] sequence) {
    if (sequence == null || sequence.length == 0)
        return false;
    return verify(sequence, 0, sequence.length - 1);
}

private boolean verify(int[] sequence, int first, int last) {
    if (last - first <= 1)
        return true;
    int rootVal = sequence[last];
    int cutIndex = first;
    while (cutIndex < last && sequence[cutIndex] <= rootVal)
        cutIndex++;
    for (int i = cutIndex; i < last; i++)
        if (sequence[i] < rootVal)
            return false;
    return verify(sequence, first, cutIndex - 1) && verify(sequence, cutIndex, last - 1);
}
```

## 32.1 从上到下打印二叉树

> 题目一：不分行从上到下打印二叉树
>
> 从上往下打印出二叉树的每个节点，同层节点从左至右打印。

```java
public class Solution {
    public ArrayList<Integer> PrintFromTopToBottom(TreeNode root) {
        ArrayList<Integer> result = new ArrayList<>();
        if (root == null) {
            return result;
        }
        LinkedList<TreeNode> nodeList = new LinkedList<>();
        nodeList.add(root);
        while (!nodeList.isEmpty()) {
            TreeNode node = nodeList.removeFirst();
            result.add(node.val);
            if (node.left != null) {
                nodeList.add(node.left);
            }
            if (node.right != null) {
                nodeList.add(node.right);
            }
        }
        return result;
    }
}
```

## 32.3 之字形打印二叉树

> 题目三：之字形打印二叉树
>
> 请实现一个函数按照之字形打印二叉树，即第一行按照从左到右的顺序打印，第二层按照从右至左的顺序打印，第三行按照从左到右的顺序打印，其他行以此类推。

```java
public ArrayList<ArrayList<Integer>> Print(TreeNode pRoot) {
    ArrayList<ArrayList<Integer>> ret = new ArrayList<>();
    Queue<TreeNode> queue = new LinkedList<>();
    queue.add(pRoot);
    boolean reverse = false;
    while (!queue.isEmpty()) {
        ArrayList<Integer> list = new ArrayList<>();
        int cnt = queue.size();
        while (cnt-- > 0) {
            TreeNode node = queue.poll();
            if (node == null)
                continue;
            list.add(node.val);
            queue.add(node.left);
            queue.add(node.right);
        }
        if (reverse)
            Collections.reverse(list);
        reverse = !reverse;
        if (list.size() != 0)
            ret.add(list);
    }
    return ret;
}
```

## 31. 栈的压入、弹出序列

> 题目：输入两个整数序列，第一个序列表示栈的压入顺序，请判断第二个序列是否可能为该栈的弹出顺序。假设压入栈的所有数字均不相等。例如序列1,2,3,4,5是某栈的压入顺序，序列4,5,3,2,1是该压栈序列对应的一个弹出序列，但4,3,5,1,2就不可能是该压栈序列的弹出序列。（注意：这两个序列的长度是相等的）

```java
public boolean IsPopOrder(int[] pushSequence, int[] popSequence) {
    int n = pushSequence.length;
    Stack<Integer> stack = new Stack<>();
    for (int pushIndex = 0, popIndex = 0; pushIndex < n; pushIndex++) {
        stack.push(pushSequence[pushIndex]);
        while (popIndex < n && !stack.isEmpty() 
                && stack.peek() == popSequence[popIndex]) {
            stack.pop();
            popIndex++;
        }
    }
    return stack.isEmpty();
}
```

## 30. 包含min函数的栈

```java
private Stack<Integer> dataStack = new Stack<>();
private Stack<Integer> minStack = new Stack<>();

public void push(int node) {
    dataStack.push(node);
    minStack.push(minStack.isEmpty() ? node : Math.min(minStack.peek(), node));
}

public void pop() {
    dataStack.pop();
    minStack.pop();
}

public int top() {
    return dataStack.peek();
}

public int min() {
    return minStack.peek();
}
```

## 29. 顺时针打印矩阵

> 题目：输入一个矩阵，按照从外向里以顺时针的顺序依次打印出每一个数字，例如，如果输入如下4 X 4矩阵： 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 则依次打印出数字1,2,3,4,8,12,16,15,14,13,9,5,6,7,11,10.

```java
public ArrayList<Integer> printMatrix(int[][] matrix) {
    ArrayList<Integer> ret = new ArrayList<>();
    int r1 = 0, r2 = matrix.length - 1, c1 = 0, c2 = matrix[0].length - 1;
    while (r1 <= r2 && c1 <= c2) {
        for (int i = c1; i <= c2; i++)
            ret.add(matrix[r1][i]);
        for (int i = r1 + 1; i <= r2; i++)
            ret.add(matrix[i][c2]);
        if (r1 != r2)
            for (int i = c2 - 1; i >= c1; i--)
                ret.add(matrix[r2][i]);
        if (c1 != c2)
            for (int i = r2 - 1; i > r1; i--)
                ret.add(matrix[i][c1]);
        r1++; r2--; c1++; c2--;
    }
    return ret;
}
```

## 28. 对称的二叉树

> 题目：请实现一个函数，用来判断一颗二叉树是不是对称的。注意，如果一个二叉树同此二叉树的镜像是同样的，定义其为对称的。

```java
boolean isSymmetrical(TreeNode pRoot) {
    if (pRoot == null)
        return true;
    return isSymmetrical(pRoot.left, pRoot.right);
}

boolean isSymmetrical(TreeNode t1, TreeNode t2) {
    if (t1 == null && t2 == null)
        return true;
    if (t1 == null || t2 == null)
        return false;
    if (t1.val != t2.val)
        return false;
    return isSymmetrical(t1.left, t2.right) && isSymmetrical(t1.right, t2.left);
}
```

## 27. 二叉树的镜像

> 题目：操作给定的二叉树，将其变换为源二叉树的镜像。

```java
public void Mirror(TreeNode root) {
    if (root == null)
        return;
    swap(root);
    Mirror(root.left);
    Mirror(root.right);
}
private void swap(TreeNode root) {
    TreeNode t = root.left;
    root.left = root.right;
    root.right = t;
}
```

## 26. 树的子结构

> 题目：输入两棵二叉树A，B，判断B是不是A的子结构。（ps：我们约定空树不是任意一个树的子结构）

```java
public boolean HasSubtree(TreeNode root1, TreeNode root2) {
    if (root1 == null || root2 == null)
        return false;
    return isSubtreeWithRoot(root1, root2) || HasSubtree(root1.left, root2) || HasSubtree(root1.right, root2);
}

private boolean isSubtreeWithRoot(TreeNode root1, TreeNode root2) {
    if (root2 == null)
        return true;
    if (root1 == null)
        return false;
    if (root1.val != root2.val)
        return false;
    return isSubtreeWithRoot(root1.left, root2.left) &&
    	   isSubtreeWithRoot(root1.right, root2.right);
}

```

## 25. 合并两个排序列表

```java
public ListNode Merge(ListNode list1, ListNode list2) {
    if (list1 == null)
        return list2;
    if (list2 == null)
        return list1;
    if (list1.val <= list2.val) {
        list1.next = Merge(list1.next, list2);
        return list1;
    } else {
        list2.next = Merge(list1, list2.next);
        return list2;
    }
}
```

## 24. 反转链表

```java
public ListNode ReverseList(ListNode head) {
    ListNode newList = new ListNode(-1);
    while (head != null) {
        ListNode next = head.next;
        head.next = newList.next;
        newList.next = head;
        head = next;
    }
    return newList.next;
}
```

## 23. 链表中环的入口节点



```java
public class Solution {

    public ListNode EntryNodeOfLoop(ListNode pHead) {
        if (pHead == null || pHead.next == null) {
            return null;
        }
        ListNode meetingNode = meetingNode(pHead);
        if (meetingNode == null) {
            return null;
        }
        // 得到环中节点的数目
        ListNode cur = meetingNode.next;
        int nodesInLoop = 1;
        while (cur != meetingNode) {
            nodesInLoop++;
            cur = cur.next;
        }
        ListNode behind = cur = pHead;
        // 先移动cur，次数为环中节点的数目
        while (nodesInLoop-- > 0) {
            cur = cur.next;
        }
        // 再移动behind和cur，相遇时即为入口节点
        while (behind != cur) {
            behind = behind.next;
            cur = cur.next;
        }
        return behind;
    }
    
    private ListNode meetingNode(ListNode pHead) {
        // 在链表中存在环时找到一快一慢两个指针相遇的节点，无环返回null
        ListNode cur = pHead.next.next, behind = pHead;
        while (cur != null) {
            if (cur == behind) {
                return cur;
            }
            if (cur.next != null) {
                cur = cur.next.next;
            } else {
                return null;
            }
            behind = behind.next;
        }
        return null;
    }
}
```





![image-20200728193111655](D:\github\ketchum2019.github.io\images\排序.png)