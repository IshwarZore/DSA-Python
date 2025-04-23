def sum_of_digits(n):
    return sum(int(digit) for digit in str(abs(n)))

def main():
    num = int(input("Enter an integer: "))
    print("Sum of digits:", sum_of_digits(num))

if __name__ == "__main__":
    main()