def reverese(l):
    s=0
    e=len(l)-1

    while s<e:
        l[s],l[e]=l[e],l[s]
        s=s+1
        e=e-1

if __name__ == "__main__":
    n=int(input("Enter the number : "))
    l=[]
    for i in range(n):
        l.append(int(input("Enter next int : ")))
    
    print(l)
    reverese(l)
    print(l)