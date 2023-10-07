class Solution:
    def lengthOfLongestSubstring(self, s: str) -> int:
        if len(s) == 0:
            return 0
        elif len(s) == 1:
            return 1
        else:
            alist = []
            for i in range(len(s)-1):
                string = s[i]
                for j in range(i+1,len(s)):
                    if s[j] not in string and j != len(s) - 1:
                        string += s[j]
                    elif s[j] not in string and j == len(s) - 1:
                        string += s[j]
                        alist.append(string)
                        break
                    else:
                        alist.append(string)
                        break
            alist = [len(alist[i]) for i in range(len(alist))]
            return max(alist)

# 官方题解
"""class Solution:
    def lengthOfLongestSubstring(self, s: str) -> int:
        # 哈希集合，记录每个字符是否出现过
        occ = set()
        n = len(s)
        # 右指针，初始值为 -1，相当于我们在字符串的左边界的左侧，还没有开始移动
        rk, ans = -1, 0
        for i in range(n):
            if i != 0:
                # 左指针向右移动一格，移除一个字符
                occ.remove(s[i - 1])
            while rk + 1 < n and s[rk + 1] not in occ:
                # 不断地移动右指针
                occ.add(s[rk + 1])
                rk += 1
            # 第 i 到 rk 个字符是一个极长的无重复字符子串
            ans = max(ans, rk - i + 1)
        return ans"""

"""class Solution:
    def lengthOfLongestSubstring(self, s: str) -> int:
        new=''
        ans=0
        for i in s:
            if i not in new:
                new=new+i
            else:
                ans=max(ans,len(new))
                idx=new.find(i)
                new=new[idx+1:]+i
                # print(idx)
        ans=max(ans,len(new))
        return ans """