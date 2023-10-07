from Python_Exercise.Data_Structure import ListNode


class Solution:
    def addTwoNumbers(self, l1, l2):
        dummy = ListNode(0)
        curr = dummy
        carry = 0

        while l1 or l2:
            if l1:
                x = l1.val
            else:
                x = 0

            if l2:
                y = l2.val
            else:
                y = 0

            sum_val = carry + x + y
            carry = sum_val // 10
            curr.next = ListNode(sum_val % 10)

            if l1:
                l1 = l1.next
            if l2:
                l2 = l2.next
            curr = curr.next

        if carry > 0:
            curr.next = ListNode(carry)

        return dummy.next
