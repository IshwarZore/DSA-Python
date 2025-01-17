def remDup(l):
    if len(l)<=1:
        return

    s=0
    while s<(len(l)-1):
        while s<(len(l)-1) and l[s]==l[s+1]:
            del l[s+1]
        s=s+1
        

if __name__ == "__main__":
    n=int(input("Enter the number : "))
    l=[]
    for i in range(n):
        l.append(int(input("Enter next int : ")))
    
    print(l)
    l.sort()
    remDup(l)
    print(l)