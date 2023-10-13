# Given an array of numbers, replace each even number with twoof the same number.
# e.g, [1,2,5,6,8, , , ,] -> [1,2,2,5,6,6,8,8].


def duplicate_even(array):

    p_end = len(array) - 1
    val_index = len(array) - 1

    while val_index >= 0:
        if array[val_index] != None:
            val_index = val_index
            break
        val_index -= 1

    while p_end >= 0:
        if array[val_index] % 2 == 0:
            array[p_end] = array[val_index]
            p_end -= 1

        array[p_end] = array[val_index]
        p_end -= 1
        val_index -= 1

    return array


print(duplicate_even([1, 2, 5, 6, 8, None, None, None]))


# Given a sentence, reverse the words of the sentence. For example,
# "i live in a house" becomes "house a in live i".


def reverse_words(sentence):

    words = sentence.split(" ")
    s, e = 0, len(words) - 1

    while s <= e:
        words[s], words[e] = words[e], words[s]
        s += 1
        e -= 1

    return " ".join(words)


print(reverse_words("i live in a house"))

# Reverse the order of elements in an array.
# For example, A = [1,2,3,4,5,6], Output = [6,5,4,3,2,1]


def reverse_numbers(array):

    s, e = 0, len(array) - 1

    while s <= e:
        array[s], array[e] = array[e], array[s]
        s += 1
        e -= 1

    return array


print(reverse_numbers([1, 2, 3, 4, 5, 6]))


# Find 2 numbers in a sorted array that sum to X.
# For example, if A = [1,2,3,4,5] and X = 9, the numbers are 4 and 5.

def find_sum(array, x):
    s, e = 0, len(array) - 1

    while s <= e:
        possible_sum = array[s] + array[e]
        if possible_sum == x:
            return array[s], array[e]
            break
        if possible_sum > x:
            e -= 1
        if possible_sum < x:
            s += 1

    return "Not found"


print(find_sum([1, 2, 3, 4, 5], 9))

# You are given an array of integers. Rearrange the array so that all zeroes are at the beginning of the array.
# For example, [4,2,0,1,0,3,0] -> [0,0,0,4,1,2,3]


def move_zeroes(array):

    fast, slow = 0, 0

    while fast < len(array):

        if array[fast] == 0:
            array[slow], array[fast] = array[fast], array[slow]
            slow += 1

        fast += 1

    return array


print(move_zeroes([4, 2, 0, 1, 0, 3, 0]))
