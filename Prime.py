def isPrime(n):
    if n==1:
        return False
    i=2
    while i*i<=n:
        if(n%i==0):
            return False
        i+=1
    return True


if __name__=="__main__":
    num=int(input("Enter the number here : "))
    print(isPrime(num))
   