def removeDuplicates(nums):
    """ Remove duplicates from sorted array in place. """

    """
    Input: nums = [0,0,1,1,1,2,2,3,3,4]
    Output: 5, nums = [0,1,2,3,4,_,_,_,_,_]

    """

    # Two pointers approach

    current = 0

    for idx in range(len(nums)):
        if nums[idx] != nums[current]:
            current += 1
            nums[current] = nums[idx]

    return current + 1  # last unique index plus one
# time complexity - O(n) because uses a single loop


def maxProfit(prices):
    """ Finds best time to buy and sell stock """

    """
    Input: prices = [7,1,5,3,6,4]
    Output: 7

    Input: prices = [1,2,3,4,5]
    Output: 4

    Input: prices = [7,6,4,3,1]
    Output: 0

    """
    max_profit = 0

    for i in range(1, len(prices)):
        if prices[i] - prices[i-1] > 0:
            max_profit += prices[i] - prices[i - 1]

    return max_profit
# time complexity - O(n)


def rotate(nums):
    """
    Do not return anything, modify nums in-place instead.
    """
    k = k % len(nums)

    nums[:] = nums[-k:] + nums[:-k]

# time complexity - O(n)


def containsDuplicate(nums):

    seen = set()

    for n in nums:
        if n in seen:
            return True
        seen.add(n)
    return False

# time complexity - O(n)


def singleNumber(nums):
    counts = {}

    for num in nums:
        if num in counts:
            counts[num] += 1
        else:
            counts[num] = 1

    for num, count in counts.items():
        if count == 1:
            return num

# time complexity - O(n)


def intersect(nums1, nums2):

    count1 = {}
    result = []

    if len(nums1) < len(nums2):
        nums1, nums2 = nums2, nums1

    for n in nums1:
        if n in count1:
            count1[n] += 1
        else:
            count1[n] = 1

    for n in nums2:
        if n in count1 and count1[n]:
            count1[n] -= 1
            result.append(n)
    return result


print(intersect([1, 4, 5, 3, 6], [2, 3, 5, 7, 9]))


def plusOne(digits):

    i = len(digits) - 1  # get index of the last digit
    digits[i] += 1  # add one to the index

    while i > 0 and digits[i] == 10:  # chechk if the digit = 10 and carry the 1
        digits[i] = 0  # set the current index to 0
        i -= 1  # move to the previos index
        digits[i] += 1  # carry and add the 1

    if digits[0] == 10:  # check if the first index = 10 so we need to create a extra digit like 999 to 1000
        digits[0] = 0  # set first index to 0
        digits.insert(0, 1)  # add a new element at index 0 with the value 1

    return digits

# time complexity - O(n)
# space complexity - O(n)


def containsDuplicate(nums):
    seen = set()

    for num in nums:
        if num in seen:
            return True
        seen.add(num)

    return False


def checkIfPangram(sentence):
    unique_elementes = set()

    for char in sentence.lower():
        if char not in unique_elementes:
            unique_elementes.add(char)

    return len(unique_elementes) == 26


def update_light(current):

    if current == 'green':
        return 'yellow'
    if current == 'yellow':
        return 'red'
    if current == 'red':
        return 'green'


def no_space(x):
    return ''.join(x.split())


def make_negative(number):
    if number > 0:
        return -number
    else:
        return number


def number_to_string(num):
    return str(num)


def bmi(weight, height):

    bmi = weight / height**2

    if bmi <= 18.5:
        return "Underweight"

    elif bmi <= 25.0:
        return "Normal"

    elif bmi <= 30.0:
        return "Overweight"

    elif bmi > 30:
        return "Obese"


def find_average(numbers):
    return sum(numbers) / len(numbers)


def better_than_average(class_points, your_points):
    return your_points > sum(class_points)/len(class_points)


def better_than_average(class_points, your_points):

    classPoints = sum(class_points, your_points)
    avg = classPoints / (len(class_points) + 1)

    return avg < your_points


def sum_array(a):

    sum = 0

    for num in a:
        sum = sum + num

    return sum


def update_light(current):

    if current == 'green':
        return 'yellow'
    if current == 'yellow':
        return 'red'
    if current == 'red':
        return 'green'


def no_space(x):
    return ''.join(x.split())


def make_negative(number):
    if number > 0:
        return -number
    else:
        return number


def number_to_string(num):
    return str(num)


def bmi(weight, height):

    bmi = weight / height**2

    if bmi <= 18.5:
        return "Underweight"

    elif bmi <= 25.0:
        return "Normal"

    elif bmi <= 30.0:
        return "Overweight"

    elif bmi > 30:
        return "Obese"


def find_average(numbers):
    return sum(numbers) / len(numbers)


def better_than_average(class_points, your_points):
    return your_points > sum(class_points)/len(class_points)


def better_than_average(class_points, your_points):

    classPoints = sum(class_points, your_points)
    avg = classPoints / (len(class_points) + 1)

    return avg < your_points


def sum_array(a):

    sum = 0

    for num in a:
        sum = sum + num

    return sum


def solution(string):

    new_string = string[::-1]

    return new_string


def multiply(a, b):
    return a * b


def missingNumber(nums):

    nums = set(nums)
    l = len(nums) + 1
    # for i in range (0, len(nums) +1):
    #     count.add(i)

    for n in (0, l):
        if n not in nums:
            return n


print(missingNumber([3, 0, 1]))


def twoSum(self, nums: List[int], target: int) -> List[int]:

    sum_nums = {}  # create an empty hash table to store seen elements

    for i, num in enumerate(nums):  # iterate through input with the index and element
        # calculate the difference between the target and the current element
        potentialMatch = target - num
        if potentialMatch in sum_nums:  # check if the potential match is in the hash table
            # if the potential match is in the hash table, return a list of the index of the potential match and the index of the current element being checked
            return [sum_nums[potentialMatch], i]
        else:  # if the potential match is not in the hash table
            # store the current element and its index in the hash table
            sum_nums[num] = i


# Input: nums = [2, 7, 11, 15], target = 9
# Output: [0, 1]


def moveZeroes(nums):
    """
        Do not return anything, modify nums in-place instead.
        """

    l = 0
    r = 0

    while r < len(nums):
        if nums[r] != 0:
            temp = nums[r]
            nums[r] = nums[l]
            nums[l] = temp
            l += 1
            r += 1
        else:
            r += 1

    return nums


# Input: nums = [0, 1, 0, 3, 12]
# Output: [1, 3, 12, 0, 0]


# For e.g, if A = [6,3,5,2,1,7]. X = 4, Result= [3,1]


def find_pair(lst, target):

    tracking = {}

    for i, num in enumerate(lst):
        if target - num in tracking:
            return [i, tracking[target - num]]
        else:
            tracking[num] = i

    return None


print(find_pair([6, 3, 5, 2, 1, 7], 10))


class Solution:
    def isAnagram(self, s: str, t: str) -> bool:

        # test
        # s = '' , t = '' >> True
        # s= 'rat1', t=' car1' >> False
        # s = "anagram2", t = "nagaram4" >> False

        # iterare, store each char on hash
        # compare s == t

        check_s = {}
        check_t = {}

        for char in s:
            if char not in check_s:
                check_s[char] = 1
            else:
                check_s[char] += 1

        for char in t:
            if char not in check_t:
                check_t[char] = 1
            else:
                check_t[char] += 1

        return check_s == check_t

# Given two strings s and t, return true if t is an anagram of s, and false otherwise.

# An Anagram is a word or phrase formed by rearranging the letters of a different word or phrase, typically using all the original letters exactly once.


# ------------ TRAVERSE ARRAY IN REVERSE

# Given an array of numbers, replace each even number with two of the same number.
# e.g, [1,2,5,6,8] -> [1,2,2,5,6,6,8,8]. Assume that the array has enough space to accommodate the result.


def duplicate_even_numbers(numbers):

    e = len(numbers) - 1
    end = len(numbers) - 1

    for i in range(e, -1, -1):
        if numbers[i] != None:
            e = i
            break

    while e >= 0:
        if numbers[e] % 2 == 0:
            numbers[end] = numbers[e]
            end -= 1

        numbers[end] = numbers[e]
        end -= 1
        e -= 1

    return numbers


print(duplicate_even_numbers([1, 2, 5, 6, 8, None, None, None]))


# Given a sentence, reverse the words of the sentence. For example, "i live in a house" becomes "house a in live i".

def reverse_words(string):
    result = []
    string_list = string.split(" ")
    end = len(string_list) - 1

    for i in range(end, -1, -1):
        result.append(string_list[i])

    return " ".join(result)


print(reverse_words("i live in a house"))

# time complexity - O(n)


# ------------ REVERSING FROM BOTH ENDS


# Reverse the order of elements in an array. For example
# A = [1,2,3,4,5,6], Output = [6,5,4,3,2,1]


def reverse_elements(a):
    start = 0
    end = len(a) - 1

    while start < end:
        a[start], a[end] = a[end], a[start]
        start += 1
        end -= 1

    return a


print(reverse_elements([1, 2, 3, 4, 5, 6, 7, 8, 9]))
print(reverse_elements([]))
print(reverse_elements([1, 2, 3]))


# time complexity -  O(n)


# Two Sum Problem - Find 2 numbers in a sorted array that sum to X.
# For example, if A = [1,2,3,4,5], and X = 9, the numbers are 4 and 5.

def find_sum(array, target):
    start = 0
    end = len(array) - 1

    while start < end:
        if array[start] + array[end] > target:
            end -= 1
        if array[start] + array[end] < target:
            start += 1
        if array[start] + array[end] == target:
            return [array[start], array[end]]

    return None


print(find_sum([1, 2, 3, 4, 5], 6))

# time complexity = O(n)


# Given a sorted array in non-decreasing order, return an array of squares of each number, also in non-decreasing order.
# For example:[-4,-2,-1,0,3,5] -> [0,1,4,9,16,25] How can you do it in O(n) time?

def number_squared(array):
    start = 0
    end = len(array) - 1

    result = []

    while start <= end:
        if abs(array[start]) > abs(array[end]):
            result.insert(0, array[start] ** 2)
            start += 1
        else:
            result.insert(0, array[end] ** 2)
            end -= 1

    return result


print(number_squared([-4, -2, -1, 0, 3, 5]))

# time complexity of O(n)


# A phrase is a palindrome if, after converting all uppercase letters into lowercase letters and removing all non-alphanumeric characters, it reads the same forward and backward. Alphanumeric characters include letters and numbers.

# Given a string s, return true if it is a palindrome, or false otherwise.


# Example 1:

# Input: s = "A man, a plan, a canal: Panama"
# Output: true
# Explanation: "amanaplanacanalpanama" is a palindrome.
# Example 2:

# Input: s = ":(start):::race a car(end)"
# Output: false
# Explanation: "raceacar" is not a palindrome.
# Example 3:

# Input: s = " "
# Output: true
# Explanation: s is an empty string "" after removing non-alphanumeric characters.
# Since an empty string reads the same forward and backward, it is a palindrome.

# "raceacar"
# "amanaplanacanalpanama"

#  s = "A man, a plan, a canal: Panama"


def find_palindrome(string):
    start = 0
    end = len(string) - 1

    while start < end:

        while start < end and not string[start].isalpha():
            start += 1
        while start < end and not string[end].isalpha():
            end -= 1
        if string[start].lower() != string[end].lower():
            return False
        start += 1
        end -= 1

    return True


print(find_palindrome(" "))

# Time Complexity: O(n)
# Space Complexity: O(1)

# ------------ PARTITIONING ARRAYS

# You are given an array of integers. Rearrange the array so that all zeroes are at the beginning of the array.
# For example, [4,2,0,1,0,3,0] -> [0,0,0,4,1,2,3]


def move_zeroes(array):
    b = 0

    for i in range(len(array)):
        if array[i] == 0:
            array[b], array[i] = array[i], array[b]
            b += 1

    return array


print(move_zeroes([0, 2, 0, 3, 1, 0, 4, 0]))


# Now, given an array, move all zeroes to the end of the array.
# For example, [4,2,0,1,0,3,0] -> [4,1,2,3,0,0,0]

def move_zeroes(array):
    b = len(array) - 1

    for i in range(len(array) - 1, -1, -1):
        if array[i] == 0:
            array[b], array[i] = array[i], array[b]
            b -= 1

    return array


print(move_zeroes([4, 2, 0, 1, 0, 3, 0]))

# ----------------keeping the order


def move_zeroes(array):
    fast = 0
    slow = 0

    while fast < len(array):
        if array[fast] != 0:
            array[slow], array[fast] = array[fast], array[slow]
            slow += 1
        fast += 1

    return array


print(move_zeroes([4, 2, 0, 1, 0, 3, 0]))

# For example, [4,2,0,1,0,3,0] -> [4,2,1,3,0,0,0]


#  Dutch National Flag Problem: Given an array of integers A and a pivot, rearrange A in the following order:
# [Elements less than pivot, elements equal to pivot, elements greater than pivot]

# For example, if A = [5,2,4,4,6,4,4,3] and pivot = 4 -> result = [3,2,4,4,4,4,6,5]

# Note: the order within each section doesn't matter.


def reorder_array(array, pivot):
    low_b = 0
    high_b = len(array) - 1
    i = 0

    while i <= high_b:
        if array[i] < pivot:
            array[i], array[low_b] = array[low_b], array[i]
            low_b += 1
            i += 1

        if array[i] > pivot:
            array[i], array[high_b] = array[high_b], array[i]
            high_b -= 1

        else:
            i += 1
    return array


print(reorder_array([5, 2, 4, 4, 6, 4, 4, 3], 4))


# ------------ SUBARRAY - SLIDING WINDOW

# Given an array of positive integers, find the contiguous subarray that sums to a given number X.

# Given an array of positive integers, find a subarray that sums to a given number X.For e.g, input = [1,2,3,5,2] and X=8, Result = [3,5] (indexes 2,3)

def find_subarray(array, target):
    s = 0
    e = 0
    sum = array[0]

    while s < len(array):
        if sum < target:
            e += 1
            sum = sum + array[e]
        elif sum > target:
            sum = sum - array[s]
            s += 1
        else:
            return (s, e)

    return None


print(find_subarray([1, 2, 3, 5, 2], 8))

# time - space complexity = O(n) / O(1)


# ------------ SUBARRAY - PREFIX SUM


# Given a String, find the longest substring with unique characters.

# For example: "whatwhywhere" --> "atwhy"

def find_longest(string):
    track = {}
    start, end, longest = 0, 0, 1
    result = [0, 0]
    track[string[0]] = 0

    while end < len(string) - 1:
        end += 1
        to_add = string[end]
        if to_add in track and track[to_add] >= start:
            start = track[to_add] + 1
        track[to_add] = end

        if end - start + 1 > longest:
            longest = end - start + 1
            result[0], result[1] = start, end

    # final = string[result[0]:result[1] + 1]  ## prints the substring

    # return final

    return result  # returns the idx


print(find_longest("whatwhywhere"))


# time - space complexity = O(n) / O(n)


# Given an array of integers, both -ve and +ve, find a contiguous subarray that sums to 0.For example: [2,4,-2,1,-3,5,-3] --> [4,-2,1,-3]


def find_sum(array, x):

    sum = 0
    map = {}

    for i in range(len(array)):
        sum += array[i]

        if sum == x:
            return (0, i)

        elif sum - x in map:
            return map[sum-1], i

        map[sum] = i

    return None


# print(find_sum([2,4,-2,1,-3,5,-3], 0))
print(find_sum([2, 4, -2, 1, -3, 5, -3], 5))
