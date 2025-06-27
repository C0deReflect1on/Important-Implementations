n = int(input())

nums = list(map(int, input().split()))
target = int(input())

left = 0
right = len(nums) - 1

def bin_search(target, left, right):
    if target < nums[left]:
        return left
    if target > nums[right]:
        return right + 1
    while left <= right:
        middle = left + (right - left) // 2
        if nums[middle] == target:
            return middle
        if nums[middle] < target:
            left = middle + 1
        else:
            right = middle - 1

    return left

print(bin_search(target, left, right))
