def reverse_with_slicing(s):
    return s[::-1]

def reverse_without_slicing(s):
    reversed_str = ""
    for char in s:
        reversed_str = char + reversed_str
    return reversed_str

def reverse_using_stack(s):
    stack = list(s)  # push all characters onto the stack
    reversed_str = ""
    while stack:
        reversed_str += stack.pop()  # pop characters from the end (LIFO)
    return reversed_str

def main():
    s = input("Enter a string: ")
    print("Reversed using slicing:        ", reverse_with_slicing(s))
    print("Reversed without slicing:      ", reverse_without_slicing(s))
    print("Reversed using stack (pop):    ", reverse_using_stack(s))

if __name__ == "__main__":
    main()