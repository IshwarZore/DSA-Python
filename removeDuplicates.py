def remove_duplicates(arr):
    return list(set(arr))

def main():
    arr = list(map(int, input("Enter numbers separated by space: ").split()))
    print("List after removing duplicates:", remove_duplicates(arr))

if __name__ == "__main__":
    main()