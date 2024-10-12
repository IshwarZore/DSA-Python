def bSearch(l,x):
    low=0
    high= len(l)-1

    while low<=high:
        mid=(low+high)//2
        if l[mid]==x:
            return mid
        elif l[mid]<x:
            low=mid+1
        elif l[mid]>x:
            high=mid-1
    return -1

def rec(l,x,s,e):
    if s>e:
        return -1
    mid=(s+e)//2
    if l[mid]==x:
        return mid
    elif l[mid]<x:
        return rec(l,x,mid+1,e)
    elif l[mid]>x:
        return rec(l,x,s,mid-1)

if __name__ == "__main__":
    n=int(input("Enter the number of elements : "))
    l=[]
    for i in range(n):
        l.append(int(input("Enter next int : ")))

    x=int(input("Enter the number to be searched : "))
    #print(bSearch(l,x))

    s=0
    e=len(l)-1
    print(rec(l,x,s,e))