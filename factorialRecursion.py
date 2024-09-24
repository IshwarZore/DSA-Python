def isFac(n):
    if n==1:
        return 1
    return n*isFac(n-1)
if __name__=="__main__":
    num=int(input("Enter the number here : "))
    print(isFac(num))
   