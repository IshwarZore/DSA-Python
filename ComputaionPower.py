def power(n,p):
    if p==0:
        return 1
    
    ##### Sol 1: Naive #####

    # if(p%2==0):
    #     return power(n,p//2)*power(n,p//2)
    # else:
    #     return power(n,p-1)*n
    
    ##### Sol 2: Recursive #####

    # temp= power(n,p//2)
    # if(p%2==0):
    #     return temp*temp
    # else:
    #     return temp*n

    #### Sol 3 : Iterative #####
    r=1
    while p>0:
        if(p%2!=0):
            r=r*n
        p=p//2
        n=n*n
    return r

if __name__ == "__main__":

    n=int(input("Enter the number :"))
    p=int(input("Enter the power :"))
    print(power(n,p))