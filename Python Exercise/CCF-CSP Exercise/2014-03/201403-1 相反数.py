def opposite_pairs(number, numbers):
    count = 0
    for num in numbers:
        if -num in numbers:
            count += 1
    return count // 2


number = int(input())
numbers = list(map(int, input().split(" ")))
result = opposite_pairs(number,numbers)
print(result)