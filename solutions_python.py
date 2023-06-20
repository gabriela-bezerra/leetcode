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


# Given a sentence, reverse the words of the sentence. For example, "i live in a house" becomes "house a in live i".


def reverse_string(string):
    sentence = string.split(' ')
    start = 0
    end = len(sentence) - 1

    while start <= end:
        sentence[start], sentence[end] = sentence[end], sentence[start]
        start += 1
        end -= 1

    return " ".join(sentence)


print(reverse_string("i live in a house maria"))

# time/space complexity - O(n) / O(n)


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


# ------------ SUBARRAY - Kadane's


# Given an integer array nums, find the subarray with the largest sum, and return its sum.

# Input: nums = [-2,1,-3,4,-1,2,1,-5,4]
# Output: 6
# Explanation: The subarray [4,-1,2,1] has the largest sum 6.

# Input: nums = [1]
# Output: 1
# Explanation: The subarray [1] has the largest sum 1.

# Input: nums = [5,4,-1,7,8]
# Output: 23
# Explanation: The subarray [5,4,-1,7,8] has the largest sum 23.

# [-2,1,-3,4,-1,2,1,-5,4]
# Input: nums = [-2,1,-3]

# curr_subarray = [4]
# _____________________________

# curr_subarray = max(2, 4)
# [100, -101, -1]

# max_subarray = max(max_array, curr_subarray)


def return_max_sum(subarray):

    max_sum = float('-inf')
    curr_sum = 0

    for i in range(len(subarray)):
        curr_sum = max(subarray[i], subarray[i] + curr_sum)

        max_sum = max(max_sum, curr_sum)

    return max_sum


# print(return_max_sum([-2,1,-3,4,-1,2,1,-5,4]))

# print(return_max_sum([1]))

# print(return_max_sum([-1]))

# print(return_max_sum([5,4,-1,7,8]))

# print(return_max_sum([100, -101, -1]))


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


# Given a sentence, reverse the words of the sentence. For example, "i live in a house" becomes "house a in live i".


def reverse_string(string):
    sentence = string.split(' ')
    start = 0
    end = len(sentence) - 1

    while start <= end:
        sentence[start], sentence[end] = sentence[end], sentence[start]
        start += 1
        end -= 1

    return " ".join(sentence)


print(reverse_string("i live in a house maria"))

# time/space complexity - O(n) / O(n)


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


# ------------ SUBARRAY - Kadane's


# Given an integer array nums, find the subarray with the largest sum, and return its sum.

# Input: nums = [-2,1,-3,4,-1,2,1,-5,4]
# Output: 6
# Explanation: The subarray [4,-1,2,1] has the largest sum 6.

# Input: nums = [1]
# Output: 1
# Explanation: The subarray [1] has the largest sum 1.

# Input: nums = [5,4,-1,7,8]
# Output: 23
# Explanation: The subarray [5,4,-1,7,8] has the largest sum 23.

# [-2,1,-3,4,-1,2,1,-5,4]
# Input: nums = [-2,1,-3]

# curr_subarray = [4]
# _____________________________

# curr_subarray = max(2, 4)
# [100, -101, -1]

# max_subarray = max(max_array, curr_subarray)


def return_max_sum(subarray):

    max_sum = float('-inf')
    curr_sum = 0

    for i in range(len(subarray)):
        curr_sum = max(subarray[i], subarray[i] + curr_sum)

        max_sum = max(max_sum, curr_sum)

    return max_sum


# print(return_max_sum([-2,1,-3,4,-1,2,1,-5,4]))

# print(return_max_sum([1]))

# print(return_max_sum([-1]))

# print(return_max_sum([5,4,-1,7,8]))

# print(return_max_sum([100, -101, -1]))


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
    track = {string[0]: 0}
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
    sum = sum + array[i]

    if sum == x:
        return (0, i)

    elif sum - x in map:  # for finding 0 use -  sum in map:
        return (map[sum-x]+1, i)

    map[sum] = i


# print(find_sum([2,4,-2,1,-3,5,-3], 0))
print(find_sum([2, 4, -2, 1, -3, 5, -3], 5))

# Given an integer array nums sorted in non-decreasing order, remove the duplicates in-place
# such that each unique element appears only once. The relative order of the elements should be kept the same. Then return the number of unique elements in nums.


def removeDuplicates(self, nums: List[int]) -> int:

    seen = set()

    b = 0

    for i in range(len(nums)):
        if nums[i] not in seen:
            seen.add(nums[i])
            nums[b] = nums[i]
            b += 1

    return b


def maxProfit(self, prices: List[int]) -> int:

    max_profit = 0

    for i in range(1, len(prices)):
        if prices[i] - prices[i-1] > 0:
            max_profit += prices[i] - prices[i-1]

    return max_profit


def move_zeroes(nums):

    b = 0

    for i in range(len(nums)):
        if nums[i] != 0:
            nums[b], nums[i] = nums[i], nums[b]
            b += 1

    return nums


print(move_zeroes([0, 1, 0, 3, 12]))


def find_sum(nums, target):

    map = {}

    for i in range(len(nums)):
        possible_sum = target - nums[i]
        if possible_sum in map:
            return [i, map[possible_sum]]
        else:
            map[nums[i]] = i

    return None


print(find_sum([2, 7, 11, 15], 9))

# Given a string s, find the first non-repeating character in it and return its index. If it does not exist, return -1.


def firstUniqChar(self, s: str) -> int:
        map = {}

        for i in range(len(s)):
            if s[i] not in map:
                map[s[i]] = 1
            else:
                map[s[i]] += 1

        for i in range(len(s)):
            if map[s[i]] == 1:
                return i
                break

        return -1



    def reverseString(self, s: List[str]) -> None:
        """
        Do not return anything, modify s in-place instead.
        """
        
        start = 0
        end = len(s) -1
        
        while start <= end:
            s[start], s[end] = s[end], s[start]
            start += 1
            end -= 1
            
        return s

def isAnagram(self, s: str, t: str) -> bool:
        
        map1 = {}
        map2 = {}
        
        for char in s:
            if char.isalpha() and char not in map1:
                map1[char] = 1
            else:
                map1[char] += 1
                
        for char in t:
            if char.isalpha() and char not in map2:
                map2[char] = 1
            else:
                map2[char] += 1

        return map1 == map2


def isPalindrome(self, s: str) -> bool:
        
        start = 0
        end = len(s) -1
        
        while start <= end:
            
            while start < end and not s[start].isalnum():
                start += 1
            while start < end and not s[end].isalnum():
                end -= 1
                
            if s[start].lower() != s[end].lower():
                return False
            start += 1
            end -= 1
        
        return True 





# ------------ LINKED LISTS



class Node:
  def __init__(self, value):
    self.value = value
    self.next = None


class LinkedList:
  def __init__(self):
    self.head = None
    self.tail = None

  def append(self, node):
    if not self.head:
      self.head = node
    else:
      self.tail.next = node
    self.tail = node

  def print_ll(self):
    curr = self.head
    
    while curr:
      print(curr.value)
      curr = curr.next


def separate_ll(L):
  if not L.head:
    return L

  curr =  L.head
  next_ = None
  odd_ll = LinkedList()
  even_ll = LinkedList()

  while curr:
    next_ = curr.next
    curr.next = None
    if curr.value % 2 == 0:
      even_ll.append(curr)
    else:
      odd_ll.append(curr)

    curr = next_

  return (odd_ll, even_ll)

# # _______create empty ll
# ll = LinkedList()

# # _______create node
# one = Node(1)
# two = Node(2)
# three = Node(3)
# five = Node(5)

# # _______add nodes to ll
# ll.append(one)
# ll.append(two)
# ll.append(three)
# ll.append(five)

# # _______call the function
# lls = separate_ll(ll)

# odd = lls[0]
# even = lls[1]

# print("ODD:::")
# odd.print_ll()

# print("EVEN::::")
# even.print_ll()


def append_ll(original, to_add):
  original.append(to_add.head)
  original.tail = to_add.tail

  return original


def sort_ll(L):

  zero = LinkedList()
  one = LinkedList()
  two = LinkedList()

  curr = L.head
  next_ = None

  while curr:
    next_ = curr.next
    curr.next = None

    if curr.value == 0:
      zero.append(curr)
    if curr.value == 1:
      one.append(curr)
    if curr.value == 2:
      two.append(curr)
    
    curr = next_

  return (zero, one, two)


ll = LinkedList()

one = Node(1)
zero = Node(0)
two = Node(2)
one_1 = Node(1)
two_1 = Node(2)
one_2 = Node(1)

ll.append(one)
ll.append(zero)
ll.append(two)
ll.append(one_1)
ll.append(two_1)
ll.append(one_2)

ll_sorted = sort_ll(ll)


appended = append_ll(ll_sorted[0], ll_sorted[1])
result = append_ll(appended, ll_sorted[2])


result.print_ll()






# There is a singly-linked list head and we want to delete a node node in it.

# Youre given the node to be deleted node. You will not be given access to the first node of head.

# All the values of the linked list are unique, and it is guaranteed that the given node node is not the last node in the linked list.

# Delete the given node. Note that by deleting the node, we do not mean removing it from memory. We mean:

# The value of the given node should not exist in the linked list.
# The number of nodes in the linked list should decrease by one.
# All the values before node should be in the same order.
# All the values after node should be in the same order.
# Custom testing:

# For the input, you should provide the entire linked list head and the node to be given node. node should not be the last node of the list and should be an actual node in the list.
# We will build the linked list and pass the node to your function.
# The output will be the entire list after calling your function.
 

# Example 1:


# Input: head = [4,5,1,9], node = 5
# Output: [4,1,9]
# Explanation: You are given the second node with value 5, the linked list should become 4 -> 1 -> 9 after calling your function.


# Example 2:

# Input: head = [4,5,1,9], node = 1
# Output: [4,5,9]
# Explanation: You are given the third node with value 1, the linked list should become 4 -> 5 -> 9 after calling your function.

# Definition for singly-linked list.
class Node:
    def __init__(self, x):
        self.val = x
        self.next = None

class LinkedList:
    def __init__(self):
        self.head = None
        self.tail = None

    def append(self, node):
        if self.head is None:
            self.head = node
        else:
            self.tail.next = node
        self.tail = node

    # Input 4 -> *5 -> 1 -> 9
    # Output 4 -> 1-> 9
    def deleteNode(self, node):
        print("Inside deleteNode::::")
        """
        :type node: ListNode
        :rtype: void Do not return anything, modify node in-place instead.
        """
        # 4 -> *5 -> 1 -> *temp1 -> 9
        node.val = node.next.val
        
        # 4 -> *1 -> 1 -> 9
        node.next = node.next.next
        # 4 -> *1 -> *temp1 -> 9
        # 4 -> 1 -> *9
        node.next.next = None

    
    def print_ll(self):
        curr = self.head
        while curr: 
            print(curr.val)
            curr = curr.next


four = Node(4)
five = Node(5)
one = Node(1)
nine = Node(9)

ll = LinkedList()

ll.append(four)
ll.append(five)
ll.append(one)
ll.append(nine)

ll.deleteNode(five)
ll.print_ll()




# Question Discussed: (Level: Medium) Print all combinations of length 3.

def print_combos(a,x):
  
  buffer = [0] * x
  print_combos_helper(a, buffer, 0, 0)

def print_combos_helper(a,buffer, a_idx, b_idx):

  if b_idx >= len(buffer):
    print(buffer)
    return
  if a_idx >= len(a):
    return

  for i in range(a_idx, len(a)):
    buffer[b_idx] = a[i]
    print_combos_helper(a, buffer, i+1, b_idx+1)


print_combos([1,2,3,4,5,6,7], 3)
  



# Level: MediumPhone Number Mnemonics: Given an N digit phone number, print all the strings that canbe made from that phone number. Since 1 and 0 don't correspond to any characters, ignorethem.For example:213 => AD, AE, AF, BD, BE, BF, CE, CE, CF456 => GJM, GJN, GJO, GKM, GKN, GKO,.. etc.



def get_letter(digit):

  if digit == 0 or digit == 1:
    return []
  elif digit == 2:
    return (["A","B", "C"])
  elif digit == 3:
    return (["D","E", "F"])
  elif digit == 4:
    return (["G","H", "I"])

  else:
    return(" Digit not valid")


def print_strings(phone):

  buffer = [""]* len(phone)
  print_strings_helper(phone, buffer, 0, 0)

def print_strings_helper(phone, buffer, p_idx, b_idx):

  if b_idx >= len(buffer) or p_idx >= len(phone):
    print(""join.buffer)
    return

  letters = get_letter(phone[p_idx])
 
  if not letters:
    print_strings_helper(phone, buffer, p_idx+1, b_idx)
  
  print("letter::", letters)
  
  for letter in letters:
    buffer[b_idx] = letter
    print_strings_helper(phone, buffer, p_idx +1, b_idx +1)


print_strings([2,1,3]) 



# Given an array of integers A, print all its subsets.
# 
# For example:Input:​ [1, 2, 3]
# Output:
# [][1][2][3][1, 2][1, 3][2, 3][1, 2, 3]


def print_sets(array):
  buffer  =  [0] * len(array)
  print_sets_helper(array, buffer, 0, 0)


def print_sets_helper(array, buffer, a_idx, b_idx):

  print(buffer[:b_idx])
  if b_idx >= len(buffer) or a_idx >= len(array):
    return

  for i in range(a_idx, len(array)):
    buffer[b_idx] = array[i]
    print_sets_helper(array, buffer, i+1, b_idx +1)


print_sets([1,2,3])

# Given an array A, print all permutations of size X.
# For example,
# Input:A = [1,2,3]X = 2
# Output:[1, 2][1, 3][2, 1][2, 3][3, 1][3, 2]


def print_perm(array, x):
  buffer = [0] * x
  is_buffer = [False] * len(array)
  print_perm_helper(array, buffer, 0, is_buffer)

def print_perm_helper(array, buffer, b_idx, is_buffer):

  if b_idx == len(buffer):
    print(buffer)
    return
    
  for i in range(len(array)):
    if not is_buffer[i]:
      buffer[b_idx] = array[i]
      is_buffer[i] = True
      print_perm_helper(array, buffer,b_idx +1, is_buffer)
      is_buffer[i] = False
      

print_perm([1,2,3], 2)


# Maze Problem: You are given a 2D array that represents a maze. It can have 2 values - 0 and 1.1 represents a wall and 0 represents a path.The objective is to reach the bottom right corner, i.e, A[A.length-1][A.length-1]. You start fromA[0][0]. You can move in 4 directions - up, down, left and right. Find if a path exists to thebottom right of the maze.For example, a path exists in the following maze:0 1 1 10 1 1 10 0 0 01 1 1 0




class State:

  UNVISETED = 0
  VISITING = 1
  NO_PATH = 2

def oob(a, i, j):

  return i < 0 or i >= len(a) or j < 0 or j >= len(a[0])

def find_path(a):

  memo = []

  for _ in range(len(a)):
    row = []
    for _ in range(len(a[0])):
      row.append(State.UNVISETED)
    memo.append(row)

  return find_path_helper(a, 0,0, memo)

def find_path_helper(a, i,j, memo):

  if oob(a,i,j) or a[i][j] == 1:
    return False

  if i == len(a) -1 and j == len(a[0]) -1:
    return True

  if memo[i][j] == State.NO_PATH or memo[i][j] == State.VISITING:
    return False

  memo[i][j] = State.VISITING

  moves = [
    (i+1, j),
    (i-1, j),
    (i, j+1),
    (i, j-1)
  ]

  for move in moves:
    if find_path_helper(a, move[0], move[1], memo):
      return True

  memo[i][j] = State.NO_PATH
  return False

print(find_path([
  [0, 1 ,1 ,1],
  [0, 1 ,1 ,1], 
  [0 ,0, 0 ,0], 
  [1 ,1, 1 ,0]
]))


# Level: MediumWordBreakProblem​: Given a String S, which contains letters and no spaces, determine if youcan break it into valid words. Return one such combination of words.You can assume that you are provided a dictionary of English words.For example:S = "ilikemangotango"Output:Return any one of these:"i like mango tango""i like man go tan go""i like mango tan go""i like man go tango"Questions to Clarify:Q. Can I return the result as a list of strings (each string is a word)?A. YesQ. What to return if no result is found?A. Just return null.Q. What if there are multiple possible results?A. Return any one valid result.Q. What if the String is empty or null?A. Return null.Solution:We use the following recursion: iterate ​i​ from 0 to the end of the string, andcheck if ​s[0..i]​is a valid word. If it’s a valid word, do the same for the remainder of the string.For example:We first iterate from 0. When ​i​ is 0, we find the first word - just "i".i​ l i k e m a n g o t a n g oWe then repeat the process with the rest of the string:result = ["i"]©​ 2017 Interview Camp (interviewcamp.io)



class STATE:
  UNVISITED = 1
  NOT_FOUND = 2

def word_break(string, dict):

  memo = [STATE.UNVISITED] * len(string)
  result = []

  if word_break_helper(string,0,result, memo,dict):
    return " ".join(result)

  return None

def word_break_helper(string,start,result, memo,dict):

  if start == len(string):
    return True

  if memo[start] == STATE.NOT_FOUND:
    return False

  for i in range(start, len(string)):
    candidate = string[start:i+1]

    if candidate in dict:
      result.append(candidate)
      if word_break_helper(string,i+1,result, memo,dict):
        return True
      else:
        result.pop()
        memo[i+1] = STATE.NOT_FOUND

  return False


DICT = {'i': 0,
        'like': 1,
        'man': 2,
        'mango': 3,
        'tan': 4,
        'tango': 5,
        'go': 6
 }

print(word_break("ilikemangotango", DICT))
    

# Find the nth number in the Fibonacci series. Fibonacci series is as follows:
# 1, 1, 2, 3, 5, 8, 13, 21, ..
# After the first two 1’s, each number is the sum of the previous two numbers.

def find_fibonacci(n):

    if n == 1 or n == 2:
      return 1

    find_fibonacci(n-1) + find_fibonacci(n-2)




# Given an array of integers, print all combinations of size X.

def print_combos(array, x):

  buffer = [0] * x

  print_combos_helper(array, buffer, 0, 0)

def print_combos_helper(array, buffer, start, buffer_idx):

  if buffer_idx == len(buffer):
    print(buffer)
    return
  if start == len(array):
    return 

  for i in range(start, len(array)):
    buffer[buffer_idx] = array[i]
    print_combos_helper(array, buffer, i+1, buffer_idx + 1)


print_combos([1,2,3,4,5,6,7], 3)


# Phone Number Mnemonics: Given an N digit phone number, print all the strings that can
# be made from that phone number. Since 1 and 0 don't correspond to any characters, ignore
# them.
# For example:
# 213 => AD, AE, AF, BD, BE, BF, CE, CE, CF
# 456 => GJM, GJN, GJO, GKM, GKN, GKO,.. etc

def  get_letter(digit):

  if digit == 1 or digit == 0:
    return [] 

  elif digit == 2:
    return ["A", "B", "C"]
  elif digit == 3:
    return ["D", "E", "F"]
  elif digit == 4:
    return ["G", "H", "I"]
  elif digit == 5:
    return ["J", "K", "L"]
  elif digit == 6:
    return ["M", "N", "O"]

  else:
    return " Digit not valid"

def print_strings(phone):

  buffer =  [''] * len(phone)
  print_strings_helpes(phone, buffer, 0,0)

def print_strings_helpes(phone, buffer, start, buffer_idx):

  if buffer_idx == len(buffer) or start == len(phone):
    print("".join(buffer))
    return

  letters = get_letter(phone[start])

  if not letters:
    print_strings_helpes(phone, buffer, start+1, buffer_idx)
    

  for letter in letters:
    buffer[buffer_idx] = letter
    print_strings_helpes(phone, buffer, start+1, buffer_idx+1)


print_strings([2,1,3])


# Given an array of integers A, print all its subsets.
# For example:
# Input:
#  [1, 2, 3]
# Output:
# []
# [1]
# [2]
# [3]
# [1, 2]
# [1, 3]
# [2, 3]
# [1, 2, 3]

def print_sets(array):

  buffer = [0] * len(array)
  print_sets_helper(array, buffer, 0,0)

def print_sets_helper(array, buffer, start, b_idx):

  print(buffer[:b_idx])

  if b_idx >= len(buffer) or start >= len(array):
    return

  for i in range(start, len(array)):
    buffer[b_idx] = array[i]
    print_sets_helper(array, buffer, i+1, b_idx+1)

print_sets([1,2,3])


# Given an array A, print all permutations of size X.
# For example,
# Input:
# A = [1,2,3]
# X = 2
# Output:
# [1, 2]
# [1, 3]
# [2, 1]
# [2, 3]
# [3, 1]
# [3, 2]


def print_perm(array, x):
  buffer = [0] * x
  is_buffer = [False] * len(array)
  print_perm_helper(array, buffer, is_buffer, 0)

def print_perm_helper(array, buffer, is_buffer, b_idx):

  if b_idx == len(buffer):
    print(buffer)
    return

  for i in range(len(array)):
    if not is_buffer[i]:
      buffer[b_idx] = array[i]
      is_buffer[i] = True
      print_perm_helper(array, buffer, is_buffer, b_idx+1)
      is_buffer[i] = False

print_perm([1,2,3], 2)


# You are given a 2D array that represents a maze. 
# It can have 2 values - 0 and 1.1 represents a wall and 0 represents a path.
# The objective is to reach the bottom right corner, i.e, A[A.length-1][A.length-1]. You start fromA[0][0]. 
# You can move in 4 directions - up, down, left and right. Find if a path exists to thebottom right of the maze.
# For example, a path exists in the following maze:
# 0 1 1 1
# 0 1 1 1
# 0 0 0 0
# 1 1 1 0


class STATE:
    UNVISETED = 0
    VISITING  = 1
    NO_PATH = 2

def find_path(array):

  memo = []

  for _ in range(len(array)):
    row = []
    for _ in range(len(array[0])):
      row.append(STATE.UNVISETED)
    memo.append(row)

  return find_path_helper(array, 0, 0, memo)

def find_path_helper(array, i, j, memo):

  if i < 0 or i >= len(array) or j < 0 or j >= len(array[0]):
    return False

  if array[i][j] == 1:
    return False

  if memo[i][j] == STATE.VISITING or memo[i][j] == STATE.NO_PATH:
    return False

  if i == len(array) -1 and j == len(array[0]) -1:
    return True

  memo[i][j] = STATE.VISITING

  points = [
    (i + 1, j),
    (i - 1, j),
    (i, j + 1),
    (i, j - 1)
  ]

  for point in points:
    if find_path_helper(array, point[0], point[1], memo):
      return True

  memo[i][j] =  STATE.NO_PATH
  return False


print(find_path([[0,1,1,1], [0,1,1,1], [0,0,0,0], [1,1,1,1]]))



def print_tuples(a):

  for i in range(len(a)):
    word = a[i]
    print ((word[0], word))


print_tuples(['house', 'car', 'doll'])

def print_tuples(numbers):

  for i in range(len(numbers)):
    if i % 2 == 0:
      print(numbers[i])
  


print_tuples([1,2,3,4,5,6,7])


def find_item(array, item):

  for i in range(len(array)):
    if array[i] == item:
      return True
  
  return False


print(find_item([1,2,3,4], 5))



def count_string(string):

  i = 0

  for char in string:
    i += 1

  
  print (i)

count_string("love")


def words_len(sentence):

  result = []

  words = sentence.split(" ")

  for word in words:
    result.append(len(word))

  return result

print(words_len("i see you tomorrow"))



def find_max(numbers):

  max = 0

  for num in numbers:
    if num > max:
      max = num

  return max


print(find_max([200,20,50,60,90,150]))


def find_max(numbers):

  track = 0
  start =  0

  while start < len(numbers):
    if numbers[start] >= track:
      track = numbers[start]
      
    start += 1

  return track

print(find_max([20,20,50,60,90,150]))


def get_longest(array):

  result = ""

  for string in array:
    if len(string) > len(result):
      result = string

  return result

print(get_longest(['hi', 'second', 'one', 'longest']))


def count_item(array, x):

  count = 0

  for item in array:
    if item == x:
      count += 1

  return count

print(count_item([1,1,1,2,5,5,], 1))

def find_indices(numbers):

  result = []

  for i in range(len(numbers)):
    if numbers[i] % 2 == 0:
      result.append(i)

  return result

print(find_indices([2,4,1]))


def replace_vowels(string):

  vowels = set("aeiou")

  chars = list(string)

  for i in range(len(chars)):
    if chars[i] in vowels:
      chars[i] = '*'

  return "".join(chars)

print(replace_vowels('returns'))


def get_unique(string):

  return list(set(string))


print(get_unique("aaaabccccddefffg"))



def count_chars(string):

  count = {}

  for char in string:
    if char in count:
      count[char] += 1
    else:
      count[char] = 1

  return count

print(count_chars('catty'))

def is_palindrome(string):

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
      

print(is_palindrome("A man, a plan, a canal: Panama"))
  


def largest_smallest(numbers, x):
  result = []
  
  for num in numbers:
    if num < x:
      result.append(num)

  return max(result)

print(largest_smallest([1,300,3, 70, 100], 100))



# Given an array of integers, print all combinations of size X.

def print_combos(array, x):

  buffer = [0] * x

  print_combos_helper(array, buffer, 0, 0)

def print_combos_helper(array, buffer, start, buffer_idx):

  if buffer_idx == len(buffer):
    print(buffer)
    return
  if start == len(array):
    return 

  for i in range(start, len(array)):
    buffer[buffer_idx] = array[i]
    print_combos_helper(array, buffer, i+1, buffer_idx + 1)


print_combos([1,2,3,4,5,6,7], 3)




# Given an array of integers, print all combinations of size X.

# Input:
# [1,2,3,4,5,6,7], 3
# Output:
# [1,2,3] [1,2,4] [1,2,5] [1,2,6] [1,2,7]
# [1,3,4] [1,3,5] [1,3,6] [1,3,7]..

# - create a new array to allocate the result - Buffer
# - find the elements to add to the buffer
# - print full buffer
# - stop once reachs end of array
# - track on the buffer and another on the array - array_index / buffer_index

# [1*,2,3,4,5,6,7]
# [0*,0,0]

def print_combos(array, x):
  buffer =  [0] * x
  print_combos_helper(array, buffer, 0, 0)

def print_combos_helper(array, buffer, array_index, buffer_index):

  if buffer_index == len(buffer):
    print(buffer)
    return

  if array_index == len(array):
    return 

  for i in range(array_index, len(array)):
    buffer[buffer_index] = array[i]
    print_combos_helper(array, buffer, i + 1, buffer_index + 1)


print_combos([1,2,3,4,5,6,7], 3)





# Phone Number Mnemonics: Given an N digit phone number, print all the strings that canbe made from that phone number. Since 1 and 0 don't correspond to any characters, ignorethem.For 

# example:
# 213 => AD, AE, AF, BD, BE, BF, CE, CE, CF
# 456 => GJM, GJN, GJO, GKM, GKN, GKO,.. etc.



# - define the valid char to each digit
# - get the char and allocate on a array
# - print full array
# - stop once reachs the end of the array
# - Ignore 1 and 0


def get_chars(digit):

  if digit == 0 or digit == 1:
    return []
  elif digit == 2:
    return ['A', 'B', 'C']
  elif digit == 3:
    return ['D', 'E', 'F']
  elif digit == 4:
    return ['G', 'H', 'I']
  elif digit == 5:
    return ['J', 'K', 'L']
  elif digit == 6:
    return ['M', 'N', 'O']


def print_phone_strings(phone):

    buffer =  [''] * len(phone)
    print_phone_strings_helper(phone, buffer, 0, 0)

def print_phone_strings_helper(phone, buffer, phone_index, buffer_index):

  if buffer_index == len(buffer) or phone_index == len(phone):
    print(''.join(buffer))
    return 

  chars =  get_chars(phone[phone_index])

  if not chars:
      print_phone_strings_helper(phone, buffer, phone_index + 1, buffer_index)

  for char in chars:
    buffer[buffer_index] = char
    print_phone_strings_helper(phone, buffer, phone_index + 1, buffer_index+1)


print_phone_strings([2,1,3])

    


def print_sets(a):

  buffer = [0] * len(a)

  print_sets_helper(a, buffer, 0, 0)

def print_sets_helper(a, buffer, a_index, buffer_index):
  
  print(buffer[:buffer_index])

  if buffer_index == len(buffer) or a_index == len(a):
    return

  for i in range(a_index, len(a)):
    buffer[buffer_index] =  a[i]
    print_sets_helper(a, buffer, i + 1, buffer_index +1)

  
print_sets([1,2,3])



def print_perm(a,x):

  buffer = [0] * x
  is_buffer = [False] * len(a)
  print_perm_helper(a, buffer, is_buffer, 0, 0)

def print_perm_helper(a, buffer, is_buffer, a_index, buffer_index):

  if buffer_index == len(buffer):
    print(buffer)
    return 

  for i in range(len(a)):
    if not is_buffer[i]:
      buffer[buffer_index] = a[i]
      is_buffer[i] = True
      print_perm_helper(a, buffer, is_buffer, i+1, buffer_index+1)
      is_buffer[i] = False

print_perm([1,2,3], 2)
      

# array = 
# [
#   [0,1,1,1]
#   [0,0,0,1]
#   [1,0,0,1]
#   [1,1,0,0]
# ]

# memo = 
# [
#   [UNVISTED],[UNVISTED],[UNVISTED],[UNVISTED]
#   [UNVISTED],[UNVISTED],[UNVISTED],[UNVISTED]
#   [UNVISTED],[UNVISTED],[UNVISTED],[UNVISTED]
#   [UNVISTED],[UNVISTED],[UNVISTED],[UNVISTED]
# ]

class State:
  UNVISITED = 0
  VISITING = 1
  NO_PATH = 2


def find_path(array):
  memo = []

  for _ in range(len(array)):
    row = []
    for _ in range (len(array[0])):
      row.append(State.UNVISITED)
    memo.append(row)

  print(find_path_helper(array, 0, 0, memo))



def find_path_helper(array, i, j, memo):

  if i < 0 or i >= len(array) or j < 0 or j >= len(array[0]):
    return False

  if array[i][j] == 1:
    return False

  if i == len(array) - 1 and j == len(array[0]) -1:
    return True

  if memo[i][j] == State.VISITING or memo[i][j] == State.NO_PATH:
    return False

  points = [
    (i +1, j),
    (i -1, j),
    (i, j +1),
    (i, j -1)
  ]
  
  memo[i][j] = State.VISITING

  for point in points:
    if find_path_helper(array, point[0], point[1], memo):
      return True
  
  memo[i][j] = State.NO_PATH
  return False

find_path( [[0,1,1,1],[0,0,0,1],[1,0,0,1],[1,1,0,0]])


VALID_WORDS = ['i', 'like','man', 'mango', 'go', 'tan', 'tango']

class State:
  UNVISITED = 0
  NOT_FOUND = 1

def words_breaker(s):

  memo = [State.UNVISITED] * len(s)
  result = []

  if words_breaker_helper(s, 0, memo, result):
    return " ".join(result)

def words_breaker_helper(s, start, memo, result):

  if start == len(s):
    return True

  if memo[start] == State.NOT_FOUND:
    return False

  for i in range(start, len(s)):
    candidate = s[start:i +1]
    if candidate in VALID_WORDS:
      result.append(candidate)
      if words_breaker_helper(s, i+1, memo, result):
        return True
      else:
        result.pop()
        memo[i] = State.NOT_FOUND

  return False

print(words_breaker("ilikemangotango"))



def words_breaker(s):

  memo = ['unvisited'] * len(s)
  result = []
  dict =  ['i', 'like', 'man', 'go', 'tan', 'go', 'mango', 'tango']

  if words_breaker_helper(s, 0, memo, result, dict):
    return ' '.join(result)

def words_breaker_helper(s, start, memo, result, dict):

  if start == len(s):
    return True

  if memo[start] == 'not_found':
    return False

  for i in range(start, len(s)):
    candidate = s[start:i+1]
    if candidate in dict:
      result.append(candidate)
      if words_breaker_helper(s, i +1, memo, result, dict):
        return True
      else:
        result.pop()
        memo[i] = 'not_found'

  return False


print(words_breaker("ilikemangotango"))


def find_largest(numbers, x):

  track = 0

  for num in numbers:
    if num < x and num > track:
      track = num 

  return track

print(find_largest([1,90,3,5,95], 100))

def is_panagram(string):

  seen = set()
  s = string.lower()

  for char in s:

    if char.isalpha() and char not in seen:
      seen.add(char)

  return len(seen) == 26

print(is_panagram('The quick brown fox jumps over the lazy dog.'))


def reverse_nums(nums):

  start = 0
  end = len(nums) -1
  
  while start <= end:
    nums[start], nums[end] = nums[end], nums[start]
    start += 1
    end -=1

  return nums

print(reverse_nums([1,2,3]))



def get_length(array):
  if not array:
    return 0
 
  return 1 + get_length(array[1:])

print(get_length([1,2,3,4,5,6]))


def print_nums(n, x):

  if n > x:
    return 

  print (n)
  print_nums(n+1, x)


print_nums(1, 5)


def max_num(num):

  curr_max = [0]
 
  max_num_helper(num, 0, curr_max)
  return curr_max
  

def max_num_helper(num, start, curr_max):

  if start == len(num):
    return

  for i in range(start, len(num)):
    if num[i] > curr_max[0]:
      curr_max[0] = num[i]
    max_num_helper(num, i+1, curr_max)
     
print(max_num([1,50,300,30]))




def max_num(num):

  curr_max = [0]
 
  max_num_helper(num, 0, curr_max)
  return curr_max[0]
  

def max_num_helper(num, start, curr_max):

  if start == len(num):
    return

  for i in range(start, len(num)):
    if num[i] > curr_max[0]:
      curr_max[0] = num[i]
    max_num_helper(num, i+1, curr_max)
     
print(max_num([1,50,300,30]))


def doubled_nums(array):

  result = []
  doubled_nums_helper(array, 0, result)
  return result 

def doubled_nums_helper(array, start, result):

  if start == len(array):
    return 

  for num in range(start, len(array)):
    result.append(num*2)
    doubled_nums_helper(array, start+1, result)

print(doubled_nums([1,2,3,4]))




# Given an m x n 2D binary grid grid which represents a map of '1's (land) and '0's (water), return the number of islands.

# An island is surrounded by water and is formed by connecting adjacent lands horizontally or vertically. You may assume all four edges of the grid are all surrounded by water.

# Example 1:

# Input: grid = [
#   ["1","1","1","1","0"],
#   ["1","1","0","1","0"],
#   ["1","1","0","0","0"],
#   ["0","0","0","0","0"]
# ]
# Output: 1
# Example 2:

# Input: grid = [
#   ["1","1","0","0","0"],
#   ["1","1","0","0","0"],
#   ["0","0","1","0","0"],
#   ["0","0","0","1","1"]
# ]
# Output: 3


# Input: grid = [
#   ["0","0","0","0","0"],
#   ["0","0","0","0","0"],
#   ["0","0","1","0","0"],
#   ["0","0","0","0","0"]
# ]
# Output: 1



# row = i
# column = j 


def count_islands(grid):

  count = 0
  for i in range(len(grid)):
    for j in range(len(grid[0])):
      if grid[i][j] == '1': 
        count +=1
        count_islands_helper(grid, 0, 0)

  return count 


def count_islands_helper(grid, i, j):
    if i < 0 or i >= len(grid) or j < 0 or j >= len(grid[0]):
      return 
    
    if grid[i][j] == '0':
      return 
  
    grid[i][j] = '0'
    
    points = [(i+1, j),(i-1,j), (i, j+1), (i, j-1)]

    for point in points:
      count_islands_helper(grid, point[0], point[1])
      
      
# grid = [
#   ["0","0","0","0","0"],
#   ["0","0","0","0","0"],
#   ["0","0","1","0","0"],
#   ["0","0","0","0","0"]
# ]  

grid2 = [
  ["1","1","0","0","0"],
  ["1","1","0","0","0"],
  ["0","0","1","0","0"],
  ["0","0","0","1","1"]
]
print(count_islands(grid2))


    
    
# bugged fixed 

# Given an m x n 2D binary grid grid which represents a map of '1's (land) and '0's (water), return the number of islands.

# An island is surrounded by water and is formed by connecting adjacent lands horizontally or vertically. You may assume all four edges of the grid are all surrounded by water.

# Example 1:

# Input: grid = [
#   ["1","1","1","1","0"],
#   ["1","1","0","1","0"],
#   ["1","1","0","0","0"],
#   ["0","0","0","0","0"]
# ]
# Output: 1
# Example 2:

# Input: grid = [
#   ["1","1","0","0","0"],
#   ["1","1","0","0","0"],
#   ["0","0","1","0","0"],
#   ["0","0","0","1","1"]
# ]
# Output: 3

# Input: grid = [
#   ["0","0","0","0","0"],
#   ["0","0","0","0","0"],
#   ["0","0","1","0","0"],
#   ["0","0","0","0","0"]
# ]
# Output: 1

# row = i
# column = j


class State:
  UNVISITED = 0
  VISITING = 1
  VISITED = 2

def count_islands(grid):

  memo = []

  for _ in range(len(grid)):
    row = []
    for _ in range(len(grid[0])):
      row.append(State.UNVISITED)
    memo.append(row)

  count = 0
  
  for i in range(len(grid)):
    for j in range(len(grid[0])):
      if grid[i][j] == '1' and memo[i][j] == State.UNVISITED :
        count += 1
        count_islands_helper(grid, i, j, memo)

  return count


def count_islands_helper(grid, i, j, memo):
  if i < 0 or i >= len(grid) or j < 0 or j >= len(grid[0]):
    return

  if grid[i][j] == '0':
    return

  if memo[i][j] == State.VISITING or memo[i][j] == State.VISITED:
    return

  memo[i][j] = State.VISITING

  points = [(i + 1, j), (i - 1, j), (i, j + 1), (i, j - 1)]

  for point in points:
    count_islands_helper(grid, point[0], point[1], memo)
  
  memo[i][j] = State.VISITED


# grid = [
#   ["0","0","0","0","0"],
#   ["0","0","0","0","0"],
#   ["0","0","1","0","0"],
#   ["0","0","0","0","0"]
# ]

grid2 = [
         ["1", "1", "0", "0", "0"], 
         ["1", "1", "0", "0", "0"],
         ["0", "0", "1", "0", "0"], 
         ["0", "0", "0", "1", "1"]
        ]

print(count_islands(grid2))


# Given an m x n 2D binary grid grid which represents a map of '1's (land) and '0's (water), return the number of islands.

# An island is surrounded by water and is formed by connecting adjacent lands horizontally or vertically. You may assume all four edges of the grid are all surrounded by water.

# Example 1:

# Input: grid = [
#   ["1","1","1","1","0"],
#   ["1","1","0","1","0"],
#   ["1","1","0","0","0"],
#   ["0","0","0","0","0"]
# ]
# Output: 1


# Input: grid2 = [
#   ["1","1","0","0","0"],
#   ["1","1","0","0","0"],
#   ["0","0","1","0","0"],
#   ["0","0","0","1","1"]
# ]
# Output: 3

#Input: grid = [
#   ["0","0","0","0","0"],
#   ["0","0","0","0","0"],
#   ["0","0","1","0","0"],
#   ["0","0","0","0","0"]
# ]
# Output: 1

# row = i
# column = j

class State:
  UNVISITED = 0
  VISITING = 1
  VISITED = 2

def count_islands(grid):

  memo = []

  for _ in range(len(grid)):
    row = []
    for _ in range( len(grid[0])):
      row.append(State.UNVISITED)
    memo.append(row)

  count = 0

  for i in range(len(grid)):
    for j in range(len(grid[0])):
      if grid[i][j] == "1" and memo[i][j] == State.UNVISITED:
        count += 1
        count_islands_helper(grid, i, j, memo)

  return count


def count_islands_helper(grid, i, j, memo):

  if i < 0 or i >= len(grid) or j < 0 or j >= len(grid[0]):
    return

  if grid[i][j] == "0":
    return 

  if memo[i][j] == State.VISITED or memo[i][j] == State.VISITING:
    return

  memo[i][j] = State.VISITING


  points = [
    (i + 1, j),
    (i - 1, j),
    (i, j + 1),
    (i, j - 1),
  ]

  for point in points:
    count_islands_helper(grid, point[0], point[1], memo)

  memo[i][j] = State.VISITED
    

# grid = [
#   ["0","0","0","0","0"],
#   ["0","0","0","0","0"],
#   ["0","0","1","0","0"],
#   ["0","0","0","0","0"]
# ]

grid = [
         ["1", "1", "0", "0", "0"], 
         ["1", "1", "0", "0", "0"],
         ["0", "0", "1", "0", "0"], 
         ["0", "0", "0", "1", "1"]
        ]

print(count_islands(grid))





def duplicate_even_numbers(array):

  start = len(array) - 1
  e = len(array) - 1

  for i in range(e, -1,-1):
    if array[i] != 0:
      e = i
      break

  while e >= 0:
    if array[e] % 2 == 0:
      array[start] =  array[e]
      start -= 1
    array[start] =  array[e]
    start -= 1
    e -= 1

  return array 

print(duplicate_even_numbers([1,2,5,6,8,0,0,0]))


def reverse_string(string):

  words = string.split(" ")

  start = 0
  end = len(words) -1

  while start <= end:
    words[start], words[end] = words[end], words[start]
    start += 1
    end -= 1

  return " ".join(words)


print(reverse_string("i live in a house"))


def find_sum(array,x):

  start = 0
  end = len(array) -1

  while start <= end:
    if array[start] + array[end] < x:
      start +=1
    if array[start] + array[end] > x:
      end -= 1    
    if array[start] + array[end] == x:
      return array[start], array[end]
      break 


print(find_sum([1,2,3,4,5], 6))

def rearrange_zeroes(array):

  b = 0
  track = 0

  while track < len(array):
    if array[track] == 0:
      array[track], array[b] = array[b],array[track]
      b += 1
      track += 1
    track +=1

  return array


print(rearrange_zeroes([4,2,0,1,0,3,0]))