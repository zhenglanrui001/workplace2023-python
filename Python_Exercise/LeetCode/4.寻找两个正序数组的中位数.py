def findMedianSortedArrays(nums1, nums2):
    alist = []
    p, q = 0, 0
    while p < len(nums1) and q < len(nums2):   # 利用归并排序 merge
        if nums1[p] < nums2[q]:
            alist.append(nums1[p])
            p += 1
        else:
            alist.append(nums2[q])
            q += 1
    alist += nums1[p:]
    alist += nums2[q:]
    if (len(nums1) + len(nums2)) % 2 != 0:
        number = (len(nums1) + len(nums2))// 2
        return float(alist[number])
    else:
        number = (len(nums1) + len(nums2)) // 2
        return float((alist[number-1] + alist[number])/2)

#print(findMedianSortedArrays([],[2,3]))