def isFac(n):
    fac=1
    for i in range(2,n+1):
       fac=fac*i
    return fac
if __name__=="__main__":
    num=int(input("Enter the number here : "))
    print(isFac(num))
   