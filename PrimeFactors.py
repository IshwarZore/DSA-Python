def PrimeFact(n):
    if n==1:
        print (n)
    i=2
    while i<=n:
        while(n%i==0):
           print (i)
           n=n//i 
        i+=1



if __name__=="__main__":
    num=int(input("Enter the number here : "))
    PrimeFact(num)
   