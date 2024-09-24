def isPal(str,s,e):
    if s>e:
        return True
    return str[s]==str[e] and isPal(str,s+1,e-1)


if __name__ == "__main__":
    str=input("Enter the string : ")
    n=len(str)-1
    print(isPal(str,0,n))