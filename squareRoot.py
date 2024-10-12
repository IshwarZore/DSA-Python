def rootFloor(n):
    i=1
    while i*i<=n:
        i=i+1
    return i-1

def rootFloorBi(n):
    s=1
    e=n
    a=-1

    while s<n:
        m=(s+e)//2

        if m*m==n:
            return m
        elif m*m>n:
            e=m-1
        elif m*m<n:
            s=m+1
            a=m                                              # if mid doesnt get updrated to a higher value of m whose 
    return a                                                 # square is smaller than n then m is the floor root

if __name__ == "__main__":
    n=int(input("Enter the number : "))
    print(rootFloor(n))