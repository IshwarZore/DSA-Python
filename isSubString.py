def is_substring(main_str, sub_str):
    return sub_str in main_str

def main():
    main_str = input("Enter the main string: ")
    sub_str = input("Enter the substring to search for: ")
    
    if is_substring(main_str, sub_str):
        print(f"'{sub_str}' is a substring of '{main_str}'.")
    else:
        print(f"'{sub_str}' is NOT a substring of '{main_str}'.")

if __name__ == "__main__":
    main()