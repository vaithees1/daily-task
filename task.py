#1
'''def longest_common_prefix(strs):
    if not strs:
        return ""
    prefix = strs[0]
    for s in strs[1:]:
        while s[:len(prefix)] != prefix:
            prefix = prefix[:-1]
            if not prefix:
                return ""
    return prefix
strs1 = ["flower", "flow", "flight"]
strs2 = ["dog", "racecar", "car"]
print(longest_common_prefix(strs1))  
print(longest_common_prefix(strs2)) 
#2
def is_prime(num):
    if num <= 1:
        return False
    for i in range(2, int(num**0.5) + 1):
        if num % i == 0:
            return False
    return True
def count_primes_in_range(start, end):
    prime_count = 0
    for num in range(start, end + 1):
        if is_prime(num):
            prime_count += 1
    return prime_count
start = int(input("Enter start of range: "))
end = int(input("Enter end of range: "))
print(f"Number of prime numbers in the range: {count_primes_in_range(start, end)}")
#3
def are_anagrams(str1, str2):
    return sorted(str1) == sorted(str2)
str1 = "listen"
str2 = "silent"
print(are_anagrams(str1, str2)) 
#4
def rotate_list(nums, k):
    k %= len(nums)
    return nums[-k:] + nums[:-k]
l = [1, 2, 3, 4, 5]
k = 3
print(rotate_list(l, k))   
#5
def highest_possible_number(n):
    return int(''.join(sorted(str(n), reverse=True)))
n = 777
print(highest_possible_number(n)) 
#6
def three_sum(nums):
    nums.sort()
    result = []
    for i in range(len(nums) - 2):
        if i > 0 and nums[i] == nums[i - 1]:
            continue
        left, right = i + 1, len(nums) - 1
        while left < right:
            total = nums[i] + nums[left] + nums[right]
            if total == 0:
                result.append([nums[i], nums[left], nums[right]])
                while left < right and nums[left] == nums[left + 1]:
                    left += 1
                while left < right and nums[right] == nums[right - 1]:
                    right -= 1
                left += 1
                right -= 1
            elif total < 0:
                left += 1
            else:
                right -= 1
    return result
nums = [-1, 0, 1, 2, -1, -4]
print(three_sum(nums))
#7
def pair_elements(a, b):
    return list(zip(a, b))
a = [1, 2, 3, 4, 5]
b = [11, 12, 13, 14, 15]
print(pair_elements(a, b))
#8
from collections import defaultdict

def group_anagrams(strs):
    anagrams = defaultdict(list)
    for s in strs:
        key = tuple(sorted(s))
        anagrams[key].append(s)
    return list(anagrams.values())
i = ["eat", "tea", "tan", "ate", "nat", "bat"]
print(group_anagrams(i))  
#9
def generate_chessboard(n=8):
    for i in range(n):
        line = ''.join(['1' if (i + j) % 2 == 0 else '0' for j in range(n)])
        print(f"{i+1}.{line}")
generate_chessboard()
#10
def is_armstrong_number(n):
    digits = list(map(int, str(n)))
    power = len(digits)
    return sum(digit ** power for digit in digits) == n
n = 153
print(is_armstrong_number(n))'''
## Day 2 task
#1
def bubble_sort(lst):
    n = len(lst)
    for i in range(n):
        for j in range(0, n-i-1):
            if lst[j] > lst[j+1]:
                lst[j], lst[j+1] = lst[j+1], lst[j]
    return lst
l = [45, 72, 82, 99, 2, 77, 8]
sorted_l = bubble_sort(l)
print(sorted_l)
#2
def is_valid_brackets(s):
    stack = []
    bracket_map = {')': '(', ']': '[', '}': '{'}
    for char in s:
        if char in bracket_map:
            top_element = stack.pop() if stack else '#'
            if bracket_map[char] != top_element:
                return False
        else:
            stack.append(char)
    return not stack

d = "{[((()))}]}}"
print(is_valid_brackets(d))
#3
def capitalize_words(s):
    return [word.capitalize() for word in s.split()]

l = "i am python welcome"
output = capitalize_words(l)
print(output)
#4
from collections import Counter

l = "i love india India"
char_count = Counter(l)
print(char_count)
#5
l = "i love india India"
char_count = Counter(l)
max_char = max(char_count, key=char_count.get)
print(max_char, char_count[max_char])

#6
def check_memory_location(lst):
    return all(x is lst[0] for x in lst)

l1 = [1, 2, 3, 4, 5]
l2 = [1, 1, 1, 1]
print(check_memory_location(l1))  
print(check_memory_location(l2))
#7
def sum_to_single_digit(n):
    while n > 9:
        n = sum(int(digit) for digit in str(n))
    return n
a = 12345
output = sum_to_single_digit(a)
print(output)
#8
from collections import Counter

input_list = [4, 6, 2, 4, 3, 4, 2, 2]
count = Counter(input_list)
output = list(count.items())
print(output)
#9
def is_palindrome(s):
    left, right = 0, len(s) - 1
    while left < right:
        if s[left] != s[right]:
            return False
        left += 1
        right -= 1
    return True

s = "racecar"
print(is_palindrome(s))
#10
def max_subarray_sum(nums):
    max_sum = float('-inf')
    current_sum = 0
    for num in nums:
        current_sum = max(num, current_sum + num)
        max_sum = max(max_sum, current_sum)
    return max_sum

nums = [-2, 1, -3, 4, -1, 2, 1, -5, 4]
output = max_subarray_sum(nums)
print(output)
### day 3 task
#1
def find_missing_number(nums):
    n = len(nums)
    total_sum = n * (n + 1) // 2
    actual_sum = sum(nums)
    return total_sum - actual_sum

nums = [3, 0, 1]
print(find_missing_number(nums))
#2
def binary_search(nums, target):
    left, right = 0, len(nums) - 1
    while left <= right:
        mid = (left + right) // 2
        if nums[mid] == target:
            return mid
        elif nums[mid] < target:
            left = mid + 1
        else:
            right = mid - 1
    return -1

nums = [1, 2, 3, 4, 5, 6, 7]
target = 5
print(binary_search(nums, target))
#3
def longest_common_subsequence(s1, s2):
    m, n = len(s1), len(s2)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if s1[i - 1] == s2[j - 1]:
                dp[i][j] = dp[i - 1][j - 1] + 1
            else:
                dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])
    return dp[m][n]

s1 = "abcdef"
s2 = "ace"
print(longest_common_subsequence(s1, s2))
#4
def rotate_matrix(matrix):
    n = len(matrix)
    for i in range(n):
        for j in range(i, n):
            matrix[i][j], matrix[j][i] = matrix[j][i], matrix[i][j]
    for row in matrix:
        row.reverse()
    return matrix

matrix = [
  [1, 2, 3],
  [4, 5, 6],
  [7, 8, 9]
]
print(rotate_matrix(matrix))
#5
def generate_parentheses(n):
    result = []
    def backtrack(s='', left=0, right=0):
        if len(s) == 2 * n:
            result.append(s)
            return
        if left < n:
            backtrack(s + '(', left + 1, right)
        if right < left:
            backtrack(s + ')', left, right + 1)
    backtrack()
    return result

n = 3
print(generate_parentheses(n))
#6
def find_peak_element(nums):
    left, right = 0, len(nums) - 1
    while left < right:
        mid = (left + right) // 2
        if nums[mid] > nums[mid + 1]:
            right = mid
        else:
            left = mid + 1
    return left

nums = [1, 2, 1, 3, 5, 6, 4]
print(find_peak_element(nums))
#7
from collections import Counter

def min_window(s, t):
    if not t or not s:
        return ""
    
    dict_t = Counter(t)
    required = len(dict_t)
    l, r = 0, 0
    formed = 0
    window_counts = {}
    ans = float("inf"), None, None

    while r < len(s):
        character = s[r]
        window_counts[character] = window_counts.get(character, 0) + 1
        
        if character in dict_t and window_counts[character] == dict_t[character]:
            formed += 1
        
        while l <= r and formed == required:
            character = s[l]
            if r - l + 1 < ans[0]:
                ans = (r - l + 1, l, r)
            window_counts[character] -= 1
            if character in dict_t and window_counts[character] < dict_t[character]:
                formed -= 1
            l += 1    
        r += 1
    return "" if ans[0] == float("inf") else s[ans[1]: ans[2] + 1]

s = "ADOBECODEBANC"
t = "ABC"
print(min_window(s, t))
#8
def next_permutation(nums):
    i = j = len(nums) - 1
    while i > 0 and nums[i - 1] >= nums[i]:
        i -= 1
    if i == 0:
        nums.reverse()
        return nums
    k = i - 1
    while nums[j] <= nums[k]:
        j -= 1
    nums[k], nums[j] = nums[j], nums[k]
    l, r = k + 1, len(nums) - 1
    while l < r:
        nums[l], nums[r] = nums[r], nums[l]
        l += 1
        r -= 1
    return nums

nums = [1, 2, 3]
print(next_permutation(nums))
#9
from collections import defaultdict

def subarray_sum(nums, k):
    count = 0
    current_sum = 0
    prefix_sums = defaultdict(int)
    prefix_sums[0] = 0

    for num in nums:
        current_sum += num
        if current_sum - k in prefix_sums:
            count += prefix_sums[current_sum - k]
        prefix_sums[current_sum] += 1

    return count

nums = [1, 1, 1]
k = 2
print(subarray_sum(nums, k))
#10
def max_area(height):
    left, right = 0, len(height) - 1
    max_area = 0

    while left < right:
        width = right - left
        max_area = max(max_area, min(height[left], height[right]) * width)
        if height[left] < height[right]:
            left += 1
        else:
            right -= 1

    return max_area

height = [1, 8, 6, 2, 5, 4, 8, 3, 7]
print(max_area(height))













