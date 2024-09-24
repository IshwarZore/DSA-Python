def isPal(n):
    rev=0
    temp=n
    while temp!=0:
        rev=rev * 10 + ( temp % 10)
        temp=temp//10
    return rev==n
if __name__=="__main__":
    num=1111
    print(isPal(num))
   