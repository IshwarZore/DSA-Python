def count_vowels(text):
    vowels = "aeiouAEIOU"
    count = 0
    for char in text:
        if char in vowels:
            count += 1
    return count

def main():
    user_input = input("Enter a string: ")
    vowel_count = count_vowels(user_input)
    print("Number of vowels:", vowel_count)

if __name__ == "__main__":
    main()