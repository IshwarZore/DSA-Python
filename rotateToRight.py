def rotate_list(lst, k):
    k = k % len(lst)  # To handle k > length of list
    return lst[-k:] + lst[:-k]

def main():
    lst = list(map(int, input("Enter list elements (space-separated): ").split()))
    k = int(input("Enter number of steps to rotate: "))
    rotated = rotate_list(lst, k)
    print("Rotated list:", rotated)

if __name__ == "__main__":
    main()