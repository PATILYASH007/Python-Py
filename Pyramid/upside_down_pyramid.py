print("\t\tPyramid\n")
n=int(input("Enter the no of rows : "))
a=input("Enter the string for pyramid :")
for i in range(n):
        for j in range(n):
            if j <i:
                print(" ", end=" ")
            else:
                print("*", end=" ")
        print()

