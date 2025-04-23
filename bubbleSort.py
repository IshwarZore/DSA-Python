def bubble_sort(arr):
    n = len(arr)
    for i in range(n):
        # Last i elements are already sorted
        for j in range(0, n - i - 1):
            if arr[j] > arr[j + 1]:
                # Swap
                arr[j], arr[j + 1] = arr[j + 1], arr[j]

def main():
    arr = list(map(int, input("Enter numbers separated by space: ").split()))
    print("Original array:", arr)
    bubble_sort(arr)
    print("Sorted array:", arr)

if __name__ == "__main__":
    main()