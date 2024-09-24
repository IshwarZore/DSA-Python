def ifSorted(l,n):
    if n<=1:
        return True
    prev=l[0]
    for i in range(1,n):
        if not l[i]>=prev:
            return False
        prev=l[i]
    return True    




if __name__ == "__main__":
    n=int(input("Enter the number : "))
    l=[]
    for i in range(n):
        l.append(int(input("Enter next int : ")))
    
    print (ifSorted(l,n))