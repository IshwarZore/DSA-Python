def fibonacci_loop(n):
    fib_seq = []
    a, b = 0, 1
    for _ in range(n):
        fib_seq.append(a)
        a, b = b, a + b
    return fib_seq

def main():
    num = int(input("Enter the number of Fibonacci terms: "))
    sequence = fibonacci_loop(num)
    print("Fibonacci sequence:")
    print(sequence)

if __name__ == "__main__":
    main()