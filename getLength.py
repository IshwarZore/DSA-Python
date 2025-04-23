def get_length(s):
    count = 0
    for _ in s:
        count += 1
    return count

def main():
    s = input("Enter a string: ")
    print("Length of the string is:", get_length(s))

if __name__ == "__main__":
    main()