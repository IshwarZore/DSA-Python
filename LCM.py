def isHCF(n,m):
    if m==0:
        return n
    return isHCF(m,n%m)
if __name__=="__main__":
    n=int(input("Enter the smaller number here : "))    #large or small does'nt matter as the two numbers swap in first recursion if 1st one is larger
    m=int(input("Enter the larger number here : "))
    print(n*m//isHCF(n,m))           
   