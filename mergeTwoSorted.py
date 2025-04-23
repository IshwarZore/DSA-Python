def merge_sorted_lists(list1, list2):
    merged = []
    i = j = 0

    # Merge while both lists have elements
    while i < len(list1) and j < len(list2):
        if list1[i] < list2[j]:
            merged.append(list1[i])
            i += 1
        else:
            merged.append(list2[j])
            j += 1

    # Add remaining elements
    merged.extend(list1[i:])
    merged.extend(list2[j:])

    return merged

def main():
    list1 = list(map(int, input("Enter sorted list 1 (space-separated): ").split()))
    list2 = list(map(int, input("Enter sorted list 2 (space-separated): ").split()))
    merged_list = merge_sorted_lists(list1, list2)
    print("Merged sorted list:", merged_list)

if __name__ == "__main__":
    main()