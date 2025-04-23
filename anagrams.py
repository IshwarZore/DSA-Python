#Anagrams are strings with same characters evil vile

def are_anagrams(str1, str2):
    # Remove spaces and convert to lowercase for case-insensitive comparison
    str1, str2 = str1.replace(" ", "").lower(), str2.replace(" ", "").lower()
    
    # Check if sorted characters of both strings are equal
    return sorted(str1) == sorted(str2)

def main():
    str1 = input("Enter the first string: ")
    str2 = input("Enter the second string: ")
    
    if are_anagrams(str1, str2):
        print("The strings are anagrams.")
    else:
        print("The strings are not anagrams.")

if __name__ == "__main__":
    main()