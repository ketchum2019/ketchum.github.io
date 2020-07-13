# Linux

## 创建进程的系统调用有那些?

**clone(), fork(), vfork();** 系统调用服务例程:sys_clone,sys_fork,sys_vfork;

## 常用命令

三大利器

grep：正则匹配查找

> 单个字符串范围符号 '[a-z]'   取反符号^	grep '[\^a]' file
>
> 边界字符：^root 头尾字符 false\$ 头尾字符 \^$空行
>
> 任意字符串: .*

![image-20200713101847885](images\image-20200713101847885.png)

sed：行编辑器 流处理编辑器（文本或管道输入）

> 

awk：文本处理工具（统计制表可编程）

> 

