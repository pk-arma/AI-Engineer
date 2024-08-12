import torch
# ================================================================ #
#             Tensor Math & Comparison Operations                  #
# ================================================================ #


x = torch.tensor([1,2,3])
y = torch.tensor([9,8,7])

# Addition
z1 = torch.empty(3)
torch.add(x,y,out=z1)
#Other way
z2 = torch.add(x,y)
z = x+y


# Subtraction
z = x - y

#  Division
z = torch.true_divide(x,y)

# Exponentiation
z = x.pow(2)
z = x**2

#Simple Multiplication
x1 = torch.rand((2,5))
x2 = torch.rand((5,3))
x3 = torch.mm(x1,x2)
x3 = x1.mm(x2)

# inplace operations
matrix_exp = torch.rand(5,5)
# print(matrix_exp.matrix_power(3))

#element wise multiplication

z= x*y
# print(x,y,z)

# dot product (The dot product, also known as the scalar product, is an algebraic operation that combines two vectors of equal length into a single number.)
z = torch.dot(x,y)

# Batch Matrix Multiplication
batch = 32
n = 10
m = 20
p = 30


tensor1 = torch.rand((batch,n,m))
# print(tensor1)
tensor2 = torch.rand((batch,m,p))
# print(tensor2)
out_bmm = torch.bmm(tensor1,tensor2) # (batch, n, p)
# print(out_bmm)

# Example of Broadcasting
x1 = torch.rand((5,5))
x2 = torch.rand((1,5))
z = x1 - x2
z = x1 ** x2

# Other useful tensor operations
sum_x = torch.sum(x,dim=0)
values, indices = torch.max(x,dim=0) #  x.max(dim=0)
values, indices = torch.min(x,dim=0)
abs_x = torch.abs(x)
z = torch.argmax(x,dim=0)
z = torch.argmax(x,dim=0)
mean_x = torch.mean(x.float(),dim=0)
z = torch.eq(x,y)
sorted_y, indices = torch.sort(y,dim=0,descending=False)
z = torch.clamp(x,min=0)
