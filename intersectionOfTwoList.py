def intersection_of_lists(list1, list2):
    return list(set(list1) & set(list2))

def main():
    list1 = list(map(int, input("Enter elements for the first list (space-separated): ").split()))
    list2 = list(map(int, input("Enter elements for the second list (space-separated): ").split()))
    
    intersection = intersection_of_lists(list1, list2)
    print("Intersection of the two lists:", intersection)

if __name__ == "__main__":
    main()